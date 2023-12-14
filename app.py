# Import streamlit for app dev
import streamlit as st
import pinecone
import os
import openai
# import langchain
from openai import OpenAI
# from langchain.embeddings.openai import OpenAIEmbeddings
from pytube import YouTube

os.environ["OPENAI_API_KEY"] = 'your key'
os.environ['PINECONE_API_KEY'] = 'your key'
os.environ['ANYSCALE_API_KEY'] = 'your key'
os.environ['ANYSCALE_API_BASE']  =  "your key"

pinecone.init(api_key=os.environ.get('PINECONE_API_KEY'), environment="gcp-starter")

def get_credentials(llm_model_name):
    if llm_model_name.lower().startswith("gpt"):
        return os.environ["OPENAI_API_BASE"], os.environ["OPENAI_API_KEY"]
    else:
        return os.environ["ANYSCALE_API_BASE"], os.environ["ANYSCALE_API_KEY"]

def generate_response(llm_model_name, temperature=0.1, stream=False, query=" ", context=" ", max_retries=3, retry_interval=60):
    '''
    Generate response from an LLM
    '''
    retry_count = 0
    system_content="""
    <<SYS>>You are a helpful assistant. Answer the query based on the context provided.
    
    The query should be answered only based on the context and 
    if context is not available don't hallucinate and return false information
    
    Also make sure to generate response based only on the context most relevant to the query
    
    Always answer as helpfully as possible, while being safe. Your answers should not include
    any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content.
    Please ensure that your responses are socially unbiased and positive in nature.
    <</SYS>>
    """
    user_content=f"query:{query} context:{context}"
    api_base, api_key = get_credentials(llm_model_name=llm_model_name)
    while retry_count < max_retries:
        try:
            client = openai.OpenAI(
                base_url = api_base,
                api_key = api_key)
            response = client.chat.completions.create(
                model=llm_model_name,
                temperature=temperature,
                stream=stream,
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content}
                ]
            )
            return response.choices[0].message.content

        except Exception as e:
            print(f"Exception: {e}")
            time.sleep(retry_interval)  # default is per-minute rate limits
            retry_count += 1
    return ""

def get_embedding(query, model_name):
    llm_name = 'llama2'
    base_url, api_key = get_credentials(llm_name)

    client = openai.OpenAI(
        base_url = base_url,
        api_key = api_key
    
    )
    query = query.replace("\n", " ")
    
    embedding = client.embeddings.create(
        model=model_name,
        input=[query]
    ).data[0].embedding
    
    return embedding

def retrieve_relevant_context(embedding, no_of_top_chunks):
    index_name = 'youtube-rag'
    index = pinecone.Index(index_name)
    relevant_chunks = index.query(embedding, top_k=no_of_top_chunks, include_metadata=True)
    context = '\n'.join([chunk['metadata']['text'] for chunk in relevant_chunks['matches']])
    titles = [chunk['metadata']['title'] for chunk in relevant_chunks['matches']]
    urls = [chunk['metadata']['url'] for chunk in relevant_chunks['matches']]
    return context,titles,urls

def rag(query, no_of_top_chunks, llm_model_name="meta-llama/Llama-2-13b-chat-hf", embedding_model_name="thenlper/gte-large", stream=False):
    embedding = get_embedding(query, embedding_model_name)
    context,titles,urls = retrieve_relevant_context(embedding, no_of_top_chunks)
    rag_response = generate_response(llm_model_name=llm_model_name,
                                    query=query,
                                    context=context)
    #rag_response = rag_response.replace('\\','')
    return rag_response,titles,urls


# Create centered main title 
st.title('🦙 Think School - RAG')
# Create a text input box for the user
prompt = st.text_input('Input your question here')

# If the user hits enter
if prompt:
    response,titles,urls = rag(query=prompt, no_of_top_chunks=5)
    thumbnail_urls = [YouTube(url).thumbnail_url for url in urls]
    titles = set(titles)
    urls = set(urls)
    thumbnail_urls = set(thumbnail_urls)
    # ...and write it out to the screen
    st.markdown(response)

    # Display raw response object
    # with st.expander('Response'):
    #     st.write(response)
    # Display source text
    with st.expander('Source'):
        st.write("The associated ThinkSchool videos are ")

        col1, col2 = st.columns(2)

        for title,url,thumbnail_url in zip(titles,urls,thumbnail_urls):
            col1.image(thumbnail_url, width=200)
            col2.markdown(f"[{title}]({url})", unsafe_allow_html=True)

        st.markdown('''
        <style>
        [data-testid="stMarkdownContainer"] ul{
            list-style-position: inside;
        }
        </style>
        ''', unsafe_allow_html=True)

st.sidebar.write("""
# About Think School

The Indian education system is messed up, and everyone knows about it. But very few are doing something to fix it. Think School is an education start-up, and we want to put a dent in the Indian education system. And we do that by providing world-class business education at less than the cost of denim jeans.

Our mission at Think School is to teach you all about business, geopolitics, and economics, which are subjects that schools and colleges typically neglect. We are here to provide an education that truly prepares you for the real world.
                 
### www.youtube.com/@ThinkSchool
""")

st.sidebar.write("""
## Disclaimer

This is a **RAG based Question Answering system** that answers questions related to Think School YouTube Channel videos.
                 
Presently, the knowledge base encompasses videos uploaded from December 2nd, 2022, up to the most recent video.
                 
This QA system can make mistakes.
""")

st.sidebar.write("""
# About me

I am Bhujith. I am a machine learning engineer. I am also someone very much interested in Business and Politics. I spend my free time watching videos related to business. ThinkSchool is one of the channels I have been following right from their beginning.
                 
The reason for developing this QA system is sometimes I would watch a video and after 2-3 days when I try to recall some key concepts I would have forgotten. 
                 
Thanks to the recent developments in GenerativeAI. This QA bot is based on Retrieval Augmented Generation (RAG). 
                 
I have chunked the videos, converted into text, and then into embeddings and stored the embeddings in a vector database. When queried, the query is converted into embedding and the text chunks corresponding to the embeddings similar to the query embedding are fetched from the vector database and provided as context to llama 2 models which generates the repsonse based on the context.
""")