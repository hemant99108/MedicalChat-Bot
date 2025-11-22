from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatResult, ChatGeneration
from typing import List, Optional
from openai import OpenAI


class PerplexityChat(BaseChatModel):

    model: str = "sonar-large-online"
    api_key: str = None

    def __init__(self, model="sonar-large-online", api_key=None):
        super().__init__()
        self.model = model
        self.api_key = api_key

        # Perplexity uses OpenAI client but different base URL
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )

    # REQUIRED by LangChain — this is where inference is done
    def _generate(self, messages: List, **kwargs) -> ChatResult:

        # Convert LangChain messages → Perplexity format
        pplx_messages = []
        for m in messages:
            if isinstance(m, HumanMessage):
                pplx_messages.append({"role": "user", "content": m.content})
            elif isinstance(m, SystemMessage):
                pplx_messages.append({"role": "system", "content": m.content})
            elif isinstance(m, AIMessage):
                pplx_messages.append({"role": "assistant", "content": m.content})

        # API call to Perplexity
        response = self.client.chat.completions.create(
            model=self.model,
            messages=pplx_messages,
        )

        content = response.choices[0].message["content"]

        generation = ChatGeneration(message=AIMessage(content=content))
        return ChatResult(generations=[generation])

    @property
    def _llm_type(self):
        return "perplexity-chat"
