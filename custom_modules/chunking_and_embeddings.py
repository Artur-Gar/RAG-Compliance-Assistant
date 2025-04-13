import numpy as np
from tqdm.notebook import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
import gc
from langchain_community.embeddings import GigaChatEmbeddings


class GetEmbsNPYBase():

    def __init__(self, 
                 list_of_texts, 
                 embedder_kwargs, 
                 chunk_max_token_size = 300, 
                 overlap_ratio = 0.1):
        '''
        Initializes the chunking and embedding class

        Args:
        list_of_texts – list of input texts to be processed  
        embedder_kwargs – arguments to configure the embedder (depend on the specific model used)  
        chunk_max_token_size – maximum number of tokens per chunk  
        overlap_ratio – fraction of tokens from the end of one chunk to include at the beginning of the next (for overlap)
        '''
        self.list_of_text = list_of_texts 
        self.chunk_max_token_size = chunk_max_token_size
        self.overlap_ratio = overlap_ratio

        self._init_model(embedder_kwargs)

    def __call__(self, kwargs): 
        self.chunk()

        return self.get_embs(kwargs), self.list_of_text, self.list_of_ids

    def chunk(self):
        '''
        The function splits the input texts into chunks
        
        Returns:
        a flat list of all chunks, while internally tracking which original text each chunk came from (list_of_ids)
        '''
        tmp = []
        self.list_of_ids = []
        for text, i in zip(self.list_of_text, range(len(self.list_of_text))):
            chunked_text = self.splitting_and_chunking(text)
            tmp.extend(chunked_text)
            self.list_of_ids.extend([i]*len(chunked_text)) # Track the source text ID for every chunk"
        self.list_of_text = tmp.copy()
        del tmp
        gc.collect()

        return self.list_of_text 

    
    def splitting_and_chunking(self, text) -> list: 
        '''
        Splits a single input text into chunks based on sentence and token count limits,
        while applying an overlap between chunks to preserve context.

        Args:
        text – the raw input text to be chunked

        Returns:
        A list of text chunks, each constrained by a maximum token length and optionally overlapping
        with the previous chunk for better continuity
        '''

        # Split the text into sentences
        splitted_text = sent_tokenize(text)

        # Since some sentences are very long, also split them by line breaks (\n)
        splitted_text_new = []
        for sentence in splitted_text:
            splitted_text_new.extend(sentence.split('\n'))
        splitted_text_new = [sentence for sentence in splitted_text_new if len(sentence)>0]
                  
        # Now fill the chunks with the text segments, applying overlap 
        chunk = ""
        overlap_text = ""
        chunks = []
        chunk_token_size = 0
        for sentence in splitted_text_new:
            sentence_tokens_count = len(word_tokenize(sentence))

            # If a sentence is too long for one chunk, finalize the current chunk, store the sentence separately, and begin a new chunk using overlap 
            if sentence_tokens_count >= self.chunk_max_token_size:
                print(f'Увеличьте максимальное количество токенов в 1 чанке. Длина куска текста {sentence_tokens_count} токенов')
                chunks.append(chunk) 
                chunk = overlap_text
                chunk_token_size = len(word_tokenize(chunk))
                overlap_text = ""

                chunks.append(sentence) 
                                
            #  If adding a new paragraph exceeds the chunk token limit, finalize the current chunk and start a new one with overlap from the previou
            elif chunk_token_size + sentence_tokens_count > self.chunk_max_token_size:
                chunks.append(chunk) 
                chunk = overlap_text
                chunk_token_size = len(word_tokenize(chunk))
                overlap_text = ""
                # Add another paragraph to the current chunk
                chunk += " " + sentence
                chunk_token_size += sentence_tokens_count

            # Continue the current chunk if the token limit is not exceeded
            elif chunk_token_size + sentence_tokens_count <= self.chunk_max_token_size:               
                chunk += " " + sentence
                chunk_token_size += sentence_tokens_count

            # Once the chunk reaches (1 - overlap_ratio) of its max size, begin storing content for overlap
            if chunk_token_size > (1-self.overlap_ratio)*self.chunk_max_token_size:
                overlap_text += " " + sentence

        # Add the final chunk to the list
        if chunk:
            chunks.append(chunk.strip())
            
        return chunks


class GetEmbsNPY_Gigachat(GetEmbsNPYBase):
    '''
    A class for initializing the Gigachat embedder via API key.

    Args:
    scope – the model to be used for embedding  
    credentials – API key for accessing Gigachat  
    verify_ssl_certs – whether to verify SSL certificates  
    timeout – request timeout duration in seconds
    '''

    def _init_model(self, kwargs={'scope': None, 
                                 'credentials': None,
                                 'verify_ssl_certs': None,
                                 'timeout': None}):
        self.model = GigaChatEmbeddings(**kwargs)

    def get_embs(self, kwargs={}):
        return np.array(self.model.embed_documents(self.list_of_text))