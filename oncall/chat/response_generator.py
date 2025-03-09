from datetime import datetime
from typing import List

from langchain_core.prompts import ChatPromptTemplate

from oncall.chat.models import ChatResponse
from oncall.lib.utils import get_llm

SYSTEM_PROMPT = "You are an AI assistant strong at customer support. Your primary function is fielding support tickets and identifying a clear description of the ticket and the time the issue occurred."

INSTRUCTION_PROMPT = f"""
You will be given the message history with a given user in a customer support channel. Your task is to form a clear description of the customer's issue and a rough timestamp of when the issue occurred. If, given the prior conversation history, you feel you have that information, you will return a SupportTicket response which has a `description` field and a `datetime` field. If you do not have enough information to return a SupportTicket, follow up with the user to get more information.

Additional Instructions:
- Be aware that the current date and time is {datetime.now().isoformat()}.
- DO not return a support ticket unless you have both a good description and a timestamp. You cannot leave timestamp unspecified.

Message History:
{{message_history}}
"""


class ResponseGenerator:
    def __init__(self, llm):
        self.llm = llm

    def generate_response(self, message_history: List[str]) -> ChatResponse:
        prompt = ChatPromptTemplate(
            [
                ("system", SYSTEM_PROMPT),
                ("human", INSTRUCTION_PROMPT),
            ]
        )
        chain = prompt | self.llm.with_structured_output(ChatResponse)
        return chain.invoke({"message_history": message_history})


if __name__ == "__main__":
    llm = get_llm("gpt-4o")
    generator = ResponseGenerator(llm)
    message_history = [
        "Hello, I'm having trouble with my account login.",
        "Can you please provide more details about the issue you're experiencing?",
        "I'm unable to login, just getting a 403 error on my page.",
        "What time did you start experiencing this issue?",
        "About 10 minutes ago.",
    ]

    response = generator.generate_response(message_history)
    print(response)