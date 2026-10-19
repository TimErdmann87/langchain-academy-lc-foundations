from dataclasses import dataclass, field
from typing import Callable, NotRequired

from dotenv import load_dotenv
from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    dynamic_prompt,
    wrap_model_call,
    wrap_tool_call,
)
from langchain.messages import ToolMessage
from langchain.tools import ToolRuntime, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

load_dotenv()


# Mock Email Client
@dataclass
class MockEmailClient:

    EMAIL_TOOLS = {"list_emails", "send_email"}

    inbox: list[dict] = field(
        default_factory=lambda: [
            {
                "id": 1,
                "from": "bob@mail.com",
                "subject": "Lunch",
                "body": "Are you free at noon?",
            },
            {
                "id": 2,
                "from": "joe@mail.com",
                "subject": "Urgent - Meeting Rescheduled",
                "body": "The meeting was moved forward to 11.00 am today.",
            },
        ]
    )
    sent: list[dict] = field(default_factory=list)

    def list_emails(self) -> list[dict]:
        return self.inbox

    def send_email(self, to: str, subject: str, body: str) -> None:
        self.sent.append({"to": to, "subject": subject, "body": body})


# static context
@dataclass
class EmailContext:
    email_client: MockEmailClient
    user_email: str
    user_password: str = "lca_course"


# mutable state
class EmailState(AgentState):
    is_auth: NotRequired[bool]


EMAIL_STATE_AUTH_KEY = "is_auth"


# Tools
@tool
def authenticate(
    password: str, runtime: ToolRuntime[EmailContext, EmailState]
) -> Command:
    """Authenticate the user with their password. Call this before any email action."""
    corr_pw = password == runtime.context.user_password
    return Command(
        update={
            EMAIL_STATE_AUTH_KEY: corr_pw,
            "messages": [
                ToolMessage(
                    (
                        "Authentication successful."
                        if corr_pw
                        else "Authentication failed: wrong password."
                    ),
                    tool_call_id=runtime.tool_call_id,
                )
            ],
        }
    )


@tool
def read_emails(runtime: ToolRuntime[EmailContext, EmailState]) -> str:
    """Read emails in the user's inbox."""
    emails = runtime.context.email_client.list_emails()
    if not emails:
        return "Inbox is empty."
    return "\n\n".join(
        f"[{m['id']}] From: {m['from']}\nSubject: {m['subject']}\n{m['body']}"
        for m in emails
    )


@tool
def send_email(
    to: str, subject: str, body: str, runtime: ToolRuntime[EmailContext, EmailState]
) -> str:
    """Send an email for the user"""
    runtime.context.email_client.send_email(to, subject, body)
    return f"Email sent to {to}."


# Middleware
@wrap_model_call
def dynamic_email_access(
    request: ModelRequest, handler: Callable[[ModelRequest], ModelResponse]
) -> ModelResponse:
    """Don't allow access to email tools if user is not authenticated."""
    if not request.state.get(EMAIL_STATE_AUTH_KEY):
        tools = [
            t
            for t in request.tools
            if getattr(t, "name", None) not in MockEmailClient.EMAIL_TOOLS
        ]
        request = request.override(tools=tools)
    return handler(request)


@wrap_tool_call
def auth_guard(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    """Defense in depth: refuse email tool calls if not authenticated, even if the model tries."""
    if request.tool_call[
        "name"
    ] in MockEmailClient.EMAIL_TOOLS and not request.state.get(EMAIL_STATE_AUTH_KEY):
        return ToolMessage(
            "Blocked: user is not authenticated.", tool_call_id=request.tool_call["id"]
        )
    return handler(request)


@dynamic_prompt
def auth_prompt(request: ModelRequest[EmailContext]) -> str:
    user = request.runtime.context.user_email
    if request.state.get(EMAIL_STATE_AUTH_KEY):
        return (
            f"You are an email assistant for {user}. The user is authenticated. "
            "You can read and send emails. Confirm recipient and content before sending."
        )
    return (
        f"You are an email assistant for {user}. The user is NOT authenticated. "
        "Ask for their password and use the tool `authenticate` before doing anything with user emails."
    )


ctx = EmailContext(user_email="tim@example.com", email_client=MockEmailClient())

agent = create_agent(
    model="gpt-5-nano",
    tools=[authenticate, read_emails, send_email],
    state_schema=EmailState,
    context_schema=ctx,
    checkpointer=InMemorySaver(),
    middleware=[
        auth_prompt,
        dynamic_email_access,
        auth_guard,
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": {"allowed_decisions": ["approve", "edit", "reject"]},
                # tools not listed (read_emails and authenticate) run without approval
            },
            description_prefix="Approve sending this email?",
        ),
    ],
)


# helper function to ask for user decision when HITL is triggered
def ask_for_decisions(interrupt_value: dict) -> list[dict]:
    decisions = []
    for action in interrupt_value["action_requests"]:
        print(f"\n--- APPROVAL NEEDED: {action['name']} ---")
        for k, v in action["args"].items():
            print(f"  {k}: {v}")
        choice = input("[a]pprove / [e]dit / [r]eject? ").strip().lower()
        if choice.startswith("a"):
            decisions.append({"type": "approve"})
        elif choice.startswith("e"):
            args = dict(action["args"])
            for k in args:
                new = input(f"  {k} [{args[k]}]: ").strip()
                if new:
                    args[k] = new
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {"name": action["name"], "args": args},
                }
            )
        else:
            reason = (
                input("  Reason (sent back to the agent): ").strip() or "User rejected."
            )
            decisions.append({"type": "reject", "message": reason})
    return decisions


# Execution
def main() -> None:
    ctx = EmailContext(user_email="tim@example.com", email_client=MockEmailClient())

    config = {"configurable": {"thread_id": "6"}}
    # config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}

    print(
        f"Email assistant ready (demo email account password: {EmailContext.user_password})."
    )

    while True:
        user_input = input("\nUser (type 'q' to quit): ")
        if user_input.strip().lower() in {"q", "quit", "exit"}:
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]}, config, context=ctx
        )

        while "__interrupt__" in result:  # paused by HumanInTheLoopMiddleware
            decisions = ask_for_decisions(result["__interrupt__"][0].value)
            result = agent.invoke(
                Command(resume={"decisions": decisions}), config, context=ctx
            )

        # print(f"\nUser: {user_input}")
        print(f"\nAssistant: {result['messages'][-1].content}")
        print(
            f"App status: (authenticated={result.get('authenticated', False)}, sent={len(ctx.email_client.sent)})"
        )
        print("\n- - - - - - - - -\n")


if __name__ == "__main__":
    main()
