from typing import Literal
import os
import re
from datetime import datetime, timedelta, timezone

from langgraph.graph import StateGraph, START, END
from langgraph.store.base import BaseStore
from langgraph.types import interrupt, Command

from email_assistant.tools import get_tools, get_tools_by_name
from email_assistant.tools.gmail.prompt_templates import GMAIL_TOOLS_PROMPT
from email_assistant.tools.gmail.gmail_tools import mark_as_read, mark_as_spam
from email_assistant.prompts import (
    triage_system_prompt,
    triage_user_prompt,
    agent_system_prompt_hitl_memory,
    default_triage_instructions,
    default_background,
    default_response_preferences,
    default_cal_preferences,
    MEMORY_UPDATE_INSTRUCTIONS,
    MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT,
)
from email_assistant.configuration import get_llm
from email_assistant.schemas import State, RouterSchema, StateInput, UserPreferences
from email_assistant.utils import parse_gmail, format_for_display, format_gmail_markdown
from email_assistant.tools.reminders import get_default_store
from dotenv import load_dotenv

load_dotenv(".env")

# Get tools with Gmail tools
tools = get_tools([
    "send_email_tool",
    "schedule_meeting_tool",
    "check_calendar_tool",
    "Question",
    "Done",
], include_gmail=True)
tools_by_name = get_tools_by_name(tools)

# Initialize the reminder store globally
reminder_store = get_default_store()


# Optional auto-accept for HITL in tests
def _maybe_interrupt(requests):
    if os.getenv("HITL_AUTO_ACCEPT", "").lower() in ("1", "true", "yes"):
        return [{"type": "accept", "args": {}}]
    return interrupt(requests)


# Safe tool invocation helper
def _safe_tool_invoke(name: str, args):
    try:
        tool = tools_by_name[name]
        return tool.invoke(args)
    except Exception as e:
        return f"Error executing {name}: {str(e)}"


# Initialize the LLM for use with router / structured output
llm = get_llm()
llm_router = llm.with_structured_output(RouterSchema)

# Initialize the LLM, enforcing tool use (of any available tools) for agent
llm = get_llm()
llm_with_tools = llm.bind_tools(tools, tool_choice="any")


def get_memory(store, namespace, default_content=None):
    """Get memory from the store or initialize with default if it doesn't exist."""
    user_preferences = store.get(namespace, "user_preferences")
    if user_preferences:
        return user_preferences.value
    else:
        store.put(namespace, "user_preferences", default_content)
        return default_content


def update_memory(store, namespace, messages):
    """Update memory profile in the store with robust fallbacks."""
    existing = store.get(namespace, "user_preferences")
    current_profile = getattr(existing, "value", str(existing) if existing else "")
    new_profile = None
    try:
        llm = get_llm().with_structured_output(UserPreferences)
        result = llm.invoke([
            {"role": "system", "content": MEMORY_UPDATE_INSTRUCTIONS.format(current_profile=current_profile, namespace=namespace)}
        ] + messages)
        new_profile = getattr(result, "user_preferences", result.get("user_preferences") if isinstance(result, dict) else None)
    except Exception as e:
        print(f"[memory] LLM update failed: {e}")
    if not new_profile:
        new_profile = current_profile
    try:
        store.put(namespace, "user_preferences", new_profile)
    except Exception as e:
        print(f"[memory] Store update failed: {e}")


# Nodes
def triage_router(state: State, store: BaseStore) -> Command[Literal["triage_interrupt_handler", "response_agent", "__end__"]]:
    """Analyze email content; create/cancel reminders and route next step."""
    
    author, to, subject, email_thread, email_id = parse_gmail(state["email_input"])

    # --- Reminder Logic: Part 1: Cancel on detected reply ---
    reply_detected = False
    user_email = os.getenv("REMINDER_NOTIFY_EMAIL")
    # Check if the author of the current email contains the user's email.
    if user_email and user_email in author:
        cancelled_count = reminder_store.cancel_reminder(thread_id=email_id)
        if cancelled_count > 0:
            print(f"🔔 Reminder cancelled for thread {email_id} because a reply from '{user_email}' was detected in the From header.")
            reply_detected = True

    user_prompt = triage_user_prompt.format(author=author, to=to, subject=subject, email_thread=email_thread)
    email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)
    triage_instructions = get_memory(store, ("email_assistant", "triage_preferences"), default_triage_instructions)
    system_prompt = triage_system_prompt.format(background=default_background, triage_instructions=triage_instructions)

    result = llm_router.invoke([{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}])
    classification = getattr(result, "classification", "respond")

    # --- Reminder Logic: Part 2: Create on Triage Decision ---
    if classification in {"respond", "notify"}:
        print(f"🔔 Classification: {classification.upper()} - This email requires attention.")
        # If a reply was detected and we cancelled an existing reminder, do NOT create a new one.
        if not reply_detected:
            default_hours = int(os.getenv("REMINDER_DEFAULT_HOURS", 48))
            due_at = datetime.now(timezone.utc) + timedelta(hours=default_hours)
            reminder_store.add_reminder(
                thread_id=email_id,
                subject=subject,
                due_at=due_at,
                reason=f"Triaged as '{classification}'"
            )
            print(f"INFO: Reminder set for thread {email_id} due at {due_at.isoformat()}")
        
        if classification == "respond":
            goto = "response_agent"
            update = {"classification_decision": classification, "messages": [{"role": "user", "content": f"Respond to the email: {email_markdown}"}]}
        else: # notify
            goto = "triage_interrupt_handler"
            update = {"classification_decision": classification}

    elif classification == "ignore":
        print(f"🚫 Classification: IGNORE - This email can be safely ignored")
        goto = END
        update = {"classification_decision": classification}
    else:
        raise ValueError(f"Invalid classification: {classification}")
    
    return Command(goto=goto, update=update)


def triage_interrupt_handler(state: State, store: BaseStore) -> Command[Literal["response_agent", "__end__"]]:
    """Handles interrupts from the triage step"""
    author, to, subject, email_thread, email_id = parse_gmail(state["email_input"])
    email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)
    messages = [{"role": "user", "content": f"Email to notify user about: {email_markdown}"}]
    request = {
        "action_request": {"action": f"Email Assistant: {state['classification_decision']}", "args": {}},
        "config": {"allow_ignore": True, "allow_respond": True, "allow_edit": False, "allow_accept": False},
        "description": email_markdown,
    }
    response = _maybe_interrupt([request])[0]

    if response["type"] == "response":
        user_input = response["args"]
        messages.append({"role": "user", "content": f"User wants to reply to the email. Use this feedback to respond: {user_input}"})
        update_memory(store, ("email_assistant", "triage_preferences"), [{"role": "user", "content": "The user decided to respond to the email, so update the triage preferences to capture this."}] + messages)
        goto = "response_agent"
    elif response["type"] == "ignore":
        messages.append({"role": "user", "content": "The user decided to ignore the email even though it was classified as notify. Update triage preferences to capture this."})
        update_memory(store, ("email_assistant", "triage_preferences"), messages)
        goto = END
    elif response["type"] == "accept":
        # Accepting a notification ends the workflow without further action
        print("INFO: User accepted notification. Ending workflow.")
        goto = END
    else:
        raise ValueError(f"Invalid response: {response}")

    return Command(goto=goto, update={"messages": messages})


def llm_call(state: State, store: BaseStore):
    """LLM decides whether to call a tool or not with Gmail-specific nudges."""
    cal_preferences = get_memory(store, ("email_assistant", "cal_preferences"), default_cal_preferences)
    response_preferences = get_memory(store, ("email_assistant", "response_preferences"), default_response_preferences)
    gmail_prompt = agent_system_prompt_hitl_memory.replace("write_email", "send_email_tool").replace("check_calendar_availability", "check_calendar_tool").replace("schedule_meeting", "schedule_meeting_tool")
    gmail_prompt += (
        "\n\nAdditional Gmail tool guidance:\n"
        "- For schedule_meeting_tool, provide start_time and end_time in ISO format (YYYY-MM-DDTHH:MM:SS).\n"
        "- Include organizer_email and attendee emails in the attendees list.\n"
        "- For check_calendar_tool, pass dates as a list of strings in DD-MM-YYYY format (e.g., ['21-05-2025']).\n"
        "- For send_email_tool, include email_id and your email_address when replying.\n"
        "- After send_email_tool, immediately call Done.\n"
    )
    try:
        author, to, subject, email_thread, email_id = parse_gmail(state.get("email_input", {}))
    except Exception:
        author = to = subject = email_thread = email_id = ""

    def extract_email(addr: str) -> str:
        if not addr:
            return ""
        if "<" in addr and ">" in addr:
            return addr.split("<")[-1].split(">")[0].strip()
        return addr.strip()

    my_email = extract_email(to)
    system_msgs = [
        {"role": "system", "content": gmail_prompt.format(tools_prompt=GMAIL_TOOLS_PROMPT, background=default_background, response_preferences=response_preferences, cal_preferences=cal_preferences)},
        {"role": "system", "content": f"Gmail context: email_id={email_id or 'NEW_EMAIL'}; my_email={my_email}"},
    ]
    prior_tool_names = []
    for m in reversed(state.get("messages", [])):
        if getattr(m, "tool_calls", None):
            try:
                prior_tool_names = [tc.get("name") for tc in m.tool_calls]
            except Exception:
                prior_tool_names = []
            break
    if "check_calendar_tool" in prior_tool_names and "schedule_meeting_tool" not in prior_tool_names and "send_email_tool" not in prior_tool_names:
        system_msgs.append({"role": "system", "content": "Now schedule the meeting with schedule_meeting_tool."})
    if "schedule_meeting_tool" in prior_tool_names and "send_email_tool" not in prior_tool_names:
        system_msgs.append({"role": "system", "content": "Now draft the reply with send_email_tool including email_id and email_address."})
    if "send_email_tool" in prior_tool_names and "Done" not in prior_tool_names:
        system_msgs.append({"role": "system", "content": "Now call Done to finalize."})

    prompt = system_msgs + state["messages"]
    try:
        msg = llm_with_tools.invoke(prompt)
    except Exception:
        msg = None
    if not getattr(msg, "tool_calls", None):
        retry = [{"role": "system", "content": "Your next output must be exactly one tool call with arguments, no assistant text."}] + prompt
        try:
            msg_retry = llm_with_tools.invoke(retry)
            if getattr(msg_retry, "tool_calls", None):
                msg = msg_retry
        except Exception:
            msg = None
    if not getattr(msg, "tool_calls", None):
        from langchain_core.messages import AIMessage
        msg = AIMessage(content="", tool_calls=[{"name": "Done", "args": {"done": True}, "id": "Done"}])
    return {"messages": [msg]}


def interrupt_handler(state: State, store: BaseStore) -> Command[Literal["llm_call", "__end__"]]:
    """Creates an interrupt for human review of tool calls"""
    result = []
    goto = "llm_call"
    for tool_call in state["messages"][-1].tool_calls:
        if tool_call["name"] not in ["send_email_tool", "schedule_meeting_tool", "Question"]:
            observation = _safe_tool_invoke(tool_call["name"], tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
            continue
        email_input = state["email_input"]
        author, to, subject, email_thread, email_id = parse_gmail(email_input)
        original_email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)
        tool_display = format_for_display(tool_call)
        description = original_email_markdown + tool_display
        config = {}
        if tool_call["name"] == "send_email_tool" or tool_call["name"] == "schedule_meeting_tool":
            config = {"allow_ignore": True, "allow_respond": True, "allow_edit": True, "allow_accept": True}
        elif tool_call["name"] == "Question":
            config = {"allow_ignore": True, "allow_respond": True, "allow_edit": False, "allow_accept": False}
        else:
            raise ValueError(f"Invalid tool call: {tool_call['name']}")

        request = {"action_request": {"action": tool_call["name"], "args": tool_call["args"]}, "config": config, "description": description}
        response = _maybe_interrupt([request])[0]

        if response["type"] == "accept":
            observation = _safe_tool_invoke(tool_call["name"], tool_call["args"])
            result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
        elif response["type"] == "edit":
            edited_args = response["args"]["args"]
            ai_message = state["messages"][-1]
            current_id = tool_call["id"]
            updated_tool_calls = [tc for tc in ai_message.tool_calls if tc["id"] != current_id] + [{"type": "tool_call", "name": tool_call["name"], "args": edited_args, "id": current_id}]
            result.append(ai_message.model_copy(update={"tool_calls": updated_tool_calls}))
            observation = _safe_tool_invoke(tool_call["name"], edited_args)
            result.append({"role": "tool", "content": observation, "tool_call_id": current_id})
            if tool_call["name"] == "send_email_tool":
                update_memory(store, ("email_assistant", "response_preferences"), [{"role": "user", "content": f"User edited the email response. Here is the initial email generated by the assistant: {tool_call['args']}. Here is the edited email: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."}])
            elif tool_call["name"] == "schedule_meeting_tool":
                update_memory(store, ("email_assistant", "cal_preferences"), [{"role": "user", "content": f"User edited the calendar invitation. Here is the initial calendar invitation generated by the assistant: {tool_call['args']}. Here is the edited calendar invitation: {edited_args}. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."}])
        elif response["type"] == "ignore":
            result.append({"role": "tool", "content": f"User ignored this {tool_call['name']} draft. Ignore this email and end the workflow.", "tool_call_id": tool_call["id"]})
            goto = END
            update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{"role": "user", "content": f"The user ignored the draft. Update triage preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."}])
        elif response["type"] == "response":
            user_feedback = response["args"]
            if tool_call["name"] == "Question":
                if any(k in str(user_feedback).lower() for k in ["spam", "phish", "junk"]):
                    email_input = state["email_input"]
                    author, to, subject, email_thread, email_id = parse_gmail(email_input)
                    original_email_markdown = format_gmail_markdown(subject, author, to, email_thread, email_id)
                    confirm = {"action_request": {"action": "mark_as_spam", "args": {"email_id": email_id}}, "config": {"allow_ignore": True, "allow_respond": False, "allow_edit": False, "allow_accept": True}, "description": original_email_markdown + "\n\nUser flagged as spam. Move this thread to Spam?"}
                    followup = _maybe_interrupt([confirm])[0]
                    if followup["type"] == "accept":
                        observation = mark_as_spam(email_id)
                        result.append({"role": "tool", "content": observation, "tool_call_id": tool_call["id"]})
                        update_memory(store, ("email_assistant", "triage_preferences"), state["messages"] + result + [{"role": "user", "content": "User marked this email as spam. Update triage preferences to classify similar emails as ignore."}])
                        goto = END
                    else:
                        result.append({"role": "tool", "content": "User declined to move to Spam.", "tool_call_id": tool_call["id"]})
                else:
                    result.append({"role": "tool", "content": f"User answered the question. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
            else:
                result.append({"role": "tool", "content": f"User gave feedback. Feedback: {user_feedback}", "tool_call_id": tool_call["id"]})
                if tool_call["name"] == "send_email_tool":
                    update_memory(store, ("email_assistant", "response_preferences"), state["messages"] + result + [{"role": "user", "content": f"User gave feedback, which we can use to update the response preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."}])
                elif tool_call["name"] == "schedule_meeting_tool":
                    update_memory(store, ("email_assistant", "cal_preferences"), state["messages"] + result + [{"role": "user", "content": f"User gave feedback, which we can use to update the calendar preferences. Follow all instructions above, and remember: {MEMORY_UPDATE_INSTRUCTIONS_REINFORCEMENT}."}])

    return Command(goto=goto, update={"messages": result})


def should_continue(state: State, store: BaseStore) -> Literal["interrupt_handler", "mark_as_read_node"]:
    """Route to tool handler, or end if Done tool called"""
    if state["messages"][-1].tool_calls:
        if any(tc["name"] == "Done" for tc in state["messages"][-1].tool_calls):
            return "mark_as_read_node"
        return "interrupt_handler"
    return "mark_as_read_node"


def mark_as_read_node(state: State):
    """Finalize Gmail flow by marking the thread as read, unless skipped."""
    skip = os.getenv("EMAIL_ASSISTANT_SKIP_MARK_AS_READ", "").lower() in ("1", "true", "yes")
    email_input = state["email_input"]
    author, to, subject, email_thread, email_id = parse_gmail(email_input)
    if skip:
        print(f"[gmail] Skipping mark_as_read for {email_id or 'UNKNOWN_ID'} (toggle enabled)")
        return
    try:
        mark_as_read(email_id)
    except Exception as e:
        print(f"[gmail] mark_as_read failed for {email_id}: {e}")


# Build workflow
agent_builder = StateGraph(State)
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("interrupt_handler", interrupt_handler)
agent_builder.add_node("mark_as_read_node", mark_as_read_node)
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges("llm_call", should_continue, {"interrupt_handler": "interrupt_handler", "mark_as_read_node": "mark_as_read_node"})
agent_builder.add_edge("mark_as_read_node", END)
response_agent = agent_builder.compile()

overall_workflow = (
    StateGraph(State, input=StateInput)
    .add_node(triage_router)
    .add_node(triage_interrupt_handler)
    .add_node("response_agent", response_agent)
    .add_node("mark_as_read_node", mark_as_read_node)
    .add_edge(START, "triage_router")
    .add_edge("mark_as_read_node", END)
)

email_assistant = overall_workflow.compile()
