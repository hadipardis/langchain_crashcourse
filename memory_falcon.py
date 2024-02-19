#!pip install langchain
from langchain import HuggingFaceHub
from langchain import PromptTemplate, LLMChain

#If you have an OpenAI API key use this section:
from langchain_openai import OpenAI

llm = OpenAI(openai_api_key="YOUR_API_KEY", temperature=0.1)

#You can also specify a specific model from OpenAI
llm = OpenAI(model_name="gpt-3.5-turbo-instruct",openai_api_key="YOUR_API_KEY", temperature=0.1)

#As an alternative, you can use an open-source LLM like Falcon or Llama from HuggingFace
#repo_id = "tiiuae/falcon-7b-instruct"
repo_id = "ericzzz/falcon-rw-1b-chat"
huggingfacehub_api_token = "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" #Use your own API key
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.1, "max_new_tokens":1500})

#Let's define our prompt using a prompt template
template="""Question: {question}
            Answer: Let's give detailed answers."""

prompt = PromptTemplate(template=template, input_variables=["question"])

#Let's create our LLM chain now
chain = LLMChain(prompt=prompt, llm=llm)

#We now run or invoke our chain and print its output response
out = chain.invoke("It today is Monday, what day is tomorrow?")

print(out["text"])

repo_id = "ericzzz/falcon-rw-1b-chat"
huggingfacehub_api_token = "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" #Use your own API key
llm = HuggingFaceHub(huggingfacehub_api_token=huggingfacehub_api_token,
                     repo_id=repo_id,
                     model_kwargs={"temperature":0.1, "max_new_tokens":1500})

from langchain.memory import ConversationBufferMemory

from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(
            "You are a nice chatbot having a conversation with a human."
        ),
        # The `variable_name` here is what must align with memory
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
)
# Notice that we `return_messages=True` to fit into the MessagesPlaceholder
# Notice that `"chat_history"` aligns with the MessagesPlaceholder name.
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

# Notice that we just pass in the `question` variables - `chat_history` gets populated by memory
out = conversation({"question": "What day is tomorrow?"})
print(out["text"])


prompt = "What is the best tourist attraction in Paris."
out = llm_model(prompt)


my_template = "What is the best tourist attraction in {city}"
prompt = PromptTemplate(template=my_template, input_variables=["city"])


out = chain("Paris")