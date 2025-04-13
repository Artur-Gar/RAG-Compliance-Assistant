import asyncio
from langchain_qdrant import Qdrant
from langchain.prompts import ChatPromptTemplate


class Talk2DB:
    def __init__(self, 
                 collection_name, 
                 embedder_inited_cls, 
                 payload_key, 
                 query_prompt=None, 
                 query_prompt_with_history=None, 
                 response_prompt=None, 
                 num_queries=3, 
                 top_k_search=3, 
                 max_context_length=6000, 
                 client_url='http://localhost:6333'):

        # Qdrant vector store built from an already existing local database
        self.vector_store = Qdrant.from_existing_collection(
            embedding=embedder_inited_cls,
            collection_name=collection_name,
            url=client_url,
            content_payload_key=payload_key
        )

        if not query_prompt:
            query_prompt = '''Используй текущее сообщение для переформулировки одного запроса для поисковой системы по семантической близости, запрос касается законодательства, поэтому должен быть формальным и хорошо обобщать запрос пользователя. Ничего лишнего не пиши, нужен запрос в систему
                                Текущее сообщение:
                                {current_message}
                                Переформулированный запрос:'''
        
        if not query_prompt_with_history:
              query_prompt_with_history = '''Используй контекст предыдущих сообщений и текущее сообщение для переформулировки одного запроса для поисковой системы по семантической близости, запрос касается законодательства, поэтому должен быть формальным и хорошо обобщать запрос пользователя. Ничего лишнего не пиши, нужен запрос в систему
                                                Контекст предыдущих сообщений и ответов:
                                                {history}
                                                Текущее сообщение:
                                                {current_message}
                                                Переформулированный запрос:'''
                
        if not response_prompt:
            response_prompt = '''Используй следующий контекст для подробного и законченного ответа на вопрос пользователя.
                                    Если из контекста никак не следует ответ на вопрос пользователя (оригинальный запрос), то сообщи об этом.
                                    Вопрос пользователя может быть очень конкретным, постарайся обобщить и найти в контексте ответ. Отвечай на русском языке.
                                    Контекст из базы данных:
                                    {retrieved_chunks}
                                    Оригинальный запрос:
                                    {original_query}
                                    Переформулированный запрос:
                                    {formulated_query}
                                    Ответ:
                                    '''
        
        # If a custom prompt template is provided, we ensure it contains the required substrings
        assert '{current_message}' in query_prompt
        assert '{history}' in query_prompt_with_history
        assert '{current_message}' in query_prompt_with_history
        assert '{retrieved_chunks}' in response_prompt
        assert '{original_query}' in response_prompt
        assert '{formulated_query}' in response_prompt

        # Converts a prompt template string into an LLM-friendly structure that also supports variable substitution in prompts' {...}
        self.formulate_query_prompt = ChatPromptTemplate.from_template(query_prompt)
        self.formulate_query_prompt_with_history = ChatPromptTemplate.from_template(query_prompt_with_history)
        response_prompt = ChatPromptTemplate.from_template(response_prompt)

        ## Retrieval part
        document_chain = StuffDocumentsChain(prompt=response_prompt, max_context_length=max_context_length)

        self.retrieval_chain = RetrievalQA(
            retriever=self.vector_store.as_retriever(search_kwargs={"k": top_k_search}), # поиск по точкам qdrant
            combine_documents_chain=document_chain
            )
        
        self.num_queries = num_queries
    
    ## Added chat history
    async def _formulate_queries(self, current_message, llm, history):
        '''
        This function generates multiple formal rephrasings of the input user query
        We use asyncio for faster generations of independent rephrasings
        
        Args:
        current_message - raw user query
        llm – language model (from LangChain)  
        history - is a list of (user, bot) message tuples: [(user1, bot1), (user2, bot2), ...]

        Returns:
        list of multiple formal versions of the query
        '''
        # given we want a few rephrasings
        if self.num_queries > 1:
            # If history is provided, it's formatted into a string
            if history:
                history_text = "\n".join([f"User: {user_msg}\nBot: {bot_resp}" for user_msg, bot_resp in history if bot_resp])
                tasks = [
                    llm.ainvoke([self.formulate_query_prompt_with_history.format(history=history_text, current_message=current_message)]) # async invoke
                    for _ in range(self.num_queries)
                    ]
            # If history is not provided
            else:
                tasks = [
                    llm.ainvoke([self.formulate_query_prompt.format(current_message=current_message)]) 
                    for _ in range(self.num_queries)
                    ]                

            responses = await asyncio.gather(*tasks) 
            return [response.content.strip() for response in responses]       
        else:
            return [current_message.strip()]


    async def get_response(self, user_query, llm, history=[]):
        '''
        The function generates multiple rephrasings of the user query (using history if available),
        runs them through the retrieval chain, and returns the final LLM answer.

        Args:
        user_query – raw user input  
        llm – language model (from LangChain)  
        history – optional list of (user, bot) message pairs

        Returns:
        answer, retrieved_chunks, original_query, selected_formulated_query
        '''
        # Reformulate the user query using prior context 
        formulated_queries = await self._formulate_queries(user_query, llm, history)
        # Submit the inputs to the LLM 
        resp, chain_input = await self.retrieval_chain(formulated_queries, user_query, llm) # resp - это combined_docs из RetrievalQA
            
        return resp['answer'], chain_input['retrieved_chunks'], chain_input['original_query'], chain_input['formulated_query']


class StuffDocumentsChain:
    def __init__(self, prompt, max_context_length=6000):
        self.prompt = prompt
        self.max_context_length = max_context_length

    # Method call in   RetrievalQA (4)
    async def __call__(self, inputs, llm):
        """
        Constructs the final prompt by combining retrieved documents and multiple reformulated queries,
        then sends it to the LLM for answer generation.

        Args:
        inputs – a dictionary containing:
        - 'retrieved_chunks': a list of retrieved documents (from RetrievalQA)
        - 'original_query': the raw user query
        - 'formulated_query': list of rephrased queries
        llm – language model (from LangChain) 

        Returns:
        A dictionary with:
        - 'retrieved_chunks': the original list of retrieved documents
        - 'answer': the LLM-generated response
        """
        # Concatenate the documents retrieved by RetrievalQA into a single string
        context = "\n".join([doc for doc in inputs['retrieved_chunks']])
        # Truncate if larger than limit
        if len(context) > self.max_context_length: 
            context = context[:self.max_context_length]

        original_query = inputs['original_query']
        # Concatenate all rephrased query versions into a single string
        formulated_query = "\n".join([fq for fq in inputs['formulated_query']]) 
        formatted_prompt = self.prompt.format(retrieved_chunks=context, original_query=original_query, formulated_query=formulated_query)
        messages = [formatted_prompt]
        response = await llm.ainvoke(messages)
        
        return {'retrieved_chunks': inputs['retrieved_chunks'], 'answer': response.content}


class RetrievalQA:
    def __init__(self, retriever, combine_documents_chain):
        self.retriever = retriever
        self.combine_documents_chain = combine_documents_chain

    async def __call__(self, formulated_queries, original_query, llm):
        """
        Retrieves relevant documents for each reformulated version of the original query,
        then passes the collected documents and queries to a document-combining chain (object of StuffDocumentsChain)

        Args:
        formulated_queries – a list of rephrased versions of the original query  
        original_query – the user's raw input  
        llm – the language model used to generate the final answer

        Returns:
        - combined_docs: the LLM-generated answer from the combined prompt  
        - chain_input: a dictionary containing the retrieved documents, original query, and reformulated queries
        """
        all_documents = []
        for f_query in formulated_queries:
            documents = self.retriever.get_relevant_documents(f_query) # Search for matching documents for every reformulated query
            for doc in documents:
                if doc.page_content not in all_documents: # Skip documents that are already in the list
                    all_documents.append(doc.page_content) 
        
        # Prepare the input for StuffDocumentsChain: retrieved documents, the original query, and its reformulated versions
        chain_input = {
            'retrieved_chunks': all_documents, 
            'original_query': original_query,
            'formulated_query': formulated_queries
        }
        combined_docs = await self.combine_documents_chain(chain_input, llm) 
        
        return combined_docs, chain_input