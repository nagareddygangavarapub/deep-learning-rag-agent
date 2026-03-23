# System Architecture
## Team:
## Date: 03/22/2026
## Members and Roles:
- Corpus Architect: Nagareddy Gangavarapu Balareddy
- Pipeline Engineer: Nagareddy Gangavarapu Balareddy
- UX Lead: Nagareddy Gangavarapu Balareddy
- Prompt Engineer: Nagareddy Gangavarapu Balareddy
- QA Lead: Nagareddy Gangavarapu Balareddy

---

## Architecture Diagram

User Query
|
v
[Streamlit UI - app.py]
|
v
[LangGraph Agent]
|
v
[query_rewrite_node] -- rewrites query for better retrieval
|
v
[retrieval_node] -- queries ChromaDB with embedded query
|
v (conditional edge: should_retry_retrieval)
|
---+-------------
|               |
generate        end
|                 |
v                 v
[generation_node] [Hallucination Guard]
|                  "No relevant content found"
v
[Groq LLM - llama-3.1-8b-instant]

The diagram must show:
- [x ] How a corpus file becomes a chunk
- [ x] How a chunk becomes an embedding
- [x ] How duplicate detection fires
- [x ] How a user query flows through LangGraph to a response
- [x ] Where the hallucination guard sits in the graph
- [x ] How conversation memory is maintained across turns

python3 << 'PYEOF'
with open('docs/architecture.md', 'r') as f:
    content = f.read()

old = '''```
User Query
    |
    v
[Streamlit UI - app.py]
    |
    v
[LangGraph Agent]
    |
    v
[query_rewrite_node] -- rewrites query for better retrieval
    |
    v
[retrieval_node] -- queries ChromaDB with embedded query
    |
    v (conditional edge: should_retry_retrieval)
    |
 ---+-------------
 |               |
generate        end
 |               |
 v               v
[generation_node]  [Hallucination Guard]
 |                  "No relevant content found"
 v
[Groq LLM - llama-3.1-8b-instant]
 |
 v
Response with Source Citations
```'''
---

## Component Descriptions

### Corpus Layer

- **Source files location:** `data/corpus/`
- which file types did your team ingest — .md, and .pdf, 

- **Landmark papers ingested:
  *Rumelhart, Hinton and Williams (1986) - Backpropagation
  - LeCun et al. (1998) - LeNet CNN
  - Hochreiter and Schmidhuber (1997) - LSTM

- **Chunking strategy:**
  512 characters with 50 character overlap using RecursiveCharacterTextSplitter.
  512 balances context richness with retrieval precision.
  50 character overlap prevents concepts spanning chunk boundaries from being lost.

- **Metadata schema:**
  *| Field | Type | Purpose |
  |---|---|---|
  | topic | string | Enables topic-based filtering in ChromaDB retrieval |
  | difficulty | string | Allows difficulty-based filtering for interview prep |
  | type | string | Describes chunk content type (concept_explanation) |
  | source | string | Source filename used for citations in responses |
  | related_topics | list | Links related concepts for cross-topic retrieval |
  | is_bonus | bool | Flags bonus topics GAN, SOM, Boltzmann |


- **Duplicate detection approach:**
  *Chunk ID is SHA-256 hash of source filename plus chunk text, truncated to 16 hex chars.
  Content hashing is more reliable than filename-based detection because it detects
  identical content even when files are renamed or re-uploaded.

- **Corpus coverage:**
  - [x ] ANN
  - [ x] CNN
  - [ x] RNN
  - [x ] LSTM
  - [ x] Seq2Seq
  - [x ] Autoencoder
  - [ ] SOM *(bonus)*
  - [ ] Boltzmann Machine *(bonus)*
  - [ ] GAN *(bonus)*

---

### Vector Store Layer

- **Database:** ChromaDB — PersistentClient
- **Local persistence path:./data/chroma_db

- **Embedding model:**
  all-MiniLM-L6-v2 via sentence-transformers local CPU

- **Why this embedding model:**
  *Local embeddings imply that corpus contents do not move out of the machine - significant in the event of.
  proprietary. all-MiniLM-L6-v2 is fast, lightweight (90MB) and performs.
  well on semantic similarity. Tradeoff is worse than OpenAI embeddings.
  but none of the cost of API and no data privacy.


- **Similarity metric:**
  *Cosine is used to measure angle between vectors hence it is not sensitive to document length change.
  Dot product would prefer to give more preference to longer documents irrespective of relevance.

- **Retrieval k:**
  * 4 chunks per query. A sufficient amount of background to provide an answer.
  at a maximum of the LLM context window of 3000 tokens.

- **Similarity threshold:**
  *Under this the guard of hallucinations shoots.
  Manual calibration using testing on topic and off topic queries on the corpus.

- **Metadata filtering:**
  *ChromaDB allows filtering by topic and level of difficulty.
  where-filter on metadata fields received by the LangGraph state.

---

### Agent Layer

- **Framework:** LangGraph

- **Graph nodes:**
  *| Node | Responsibility |
  |---|---|
  | query_rewrite_node | Rewrites natural language query into keyword-dense search query for better retrieval |
  | retrieval_node | Queries ChromaDB with embedded query and returns top-k relevant chunks |
  | generation_node | Generates answer from retrieved context using Groq LLM with source citations |


- **Conditional edges:**
Checks upon retrieval After retrievalnode, shouldretryretrieval checks nocontextfound flag.
Guard fires hallucination and True routes to END.
Answer generation False False node to generationnode.

- **Hallucination guard:**
  *Returns: I could not find any information in the corpus on your query.
This could be because the subject has not     been taught yet in the course of study, or your question.
may need to be rephrased. Please consider a more narrow deep learning topic.*

- **Query rewriting:**
  *- Raw query: I do not understand the way LSTMs store information in the long-term.
- Redisputed query: LSTM long-term memory cell state forget gate mechanism.

- **Conversation memory:**
  *MemorySaver checkpointer keeps up history of threadid across turns.
trimmessages Trunchest oldest messages when history is greater than MAXCONTEXTTOKENS 3000.
Histories are lost when the app restarts as MemorySaver is only in memory.

- **LLM provider:Groq - llama-3.1-8b-instant model.
  
- **Why this provider:**
  *Free tier and no local GPU requirement and extremely low latency through Groq LPU chip.
Groq operates a custom Language Processing Unit with lower latency than the inference with the GPU.

---

### Prompt Layer

- **System prompt summary:**
  *only answer given through context, always reference with.
 when context is inadequate, modify.
technical depth technical- metadata difficulty level.

- **Question generation prompt:**
  *difficulty level and text of context chunk.
Answers: Question, difficulty, topic, modelanswer, followup, sourcecitations (JSON).
Respond with: the JSON object only. 

- **Answer evaluation prompt:**
  *Inputs: question, candidateanswer, context ground truth.
Returns: 0-10, whatwascorrect, whatwasmissing, idealanswer, JSON.
interviewverdict (hire/consider/no hire), coachingtip.
Scoring rubric: 9-10 fully, 7-8 mostly, 5-6 core, 3-4 partially, 0-2 fundamentally.

- **JSON reliability:**
  *(Added Response with the JSON only. No preamble, explanation, or code fences.
to ask generation questions as well as answer evaluation questions.

- **Failure modes identified:**
  *- System prompt: model can employ general knowledge - moderated by strictly answering ONLY of context instruction.
- Question generation: can give out malformed JSON - alleviated by the explicit instruction of JSON only.
- Answer rating: scored excessively generously - alleviated through elaborate scoring rubric in prompt

---

### Interface Layer

- **Framework:** Streamlit 
- **Deployment platform:** Streamlit Community Cloud 
- **Public URL:** [(https://deep-learning-rag-agent-bwzbxisrnomkj3tjg3aula.streamlit.app/)](http://localhost:8504/)

- **Ingestion panel features:**
  Sidebar PDF and MD multi-file uploader. Ingest Documents button activates.
DocumentChunker chunking, VectorStoreManager embedding. Shows success message
with number of chunks, duplicate warning, failure error. Recalls the list of documents consumed.
including the name of the source, subject and the number of chunks and delete button per document.

- **Document viewer features:**
  Displays no documents when no documents ingested. Complete enforcement would demonstrate
metadata badged document list and chunk content viewer.

- **Chat panel features:**
 Chat with history of persistent messages. The citations of sources are presented in expandable.
Each response will be followed by a sources section below it. Hallucination guard came up as yellow warning banner.
Chat box is fixed at the bottom of the panel.

- **Session state keys:**

  | Key | Stores |
  |---|---|
  | chat_history | |
  | ingested_documents | |
  | selected_document | |
  | thread_id | |

- **Stretch features implemented:**
  No - everything core is ready and functional.

---

## Design Decisions

Document at least three deliberate decisions your team made.
These are your Hour 3 interview talking points — be specific.
"We used the default settings" is not a design decision.

1. **Decision:**
   Content-addressed chunk IDs by computing a SHA-256 hash of text and source.
rationale Filename-based IDs cannot withstand file renaming or uploading a new file.
True duplicates are found irrespective of changes in the names of the files.
Interview response: This is generated by hashing up the source file name and chunk text.
with SHA-256. This implies that uploading the same content twice would always give the same ID.
performing the detection of duplicates with high accuracy, even with renaming files.

2. **Decision:API embeddings vs. local embeddings with all-MiniLM-L6-v2.
Rationale: API is free, no data is sent out of the machine, rate or latency is not limited.
Compared to OpenAI text-embedding-3-small, tradeoff is somewhat of lower quality.
Interview response: We opted to do local embeddings since the content of the corpus does not go beyond the machine.
that is essential to proprietary training data. The tradeoff of quality is agreeable.
all-MiniLM-L6-v2 is a strong model on semantic similarity problems.

3. **Decision:Cosine similarity with hallucination guard threshold of 0.3.
Reason: Cosine is also resistant to change in the length of the document as compared to dot product.
0.3 manual threshold - not too big to pick pertinent chunks,
high enough to disregard off topic questions such as history of Rome.
Interview response: Hallucination guard shoots when no chunk has a score above 0.3.
cosine similarity. Instead of having the LLM respond to parametric memory.
we send an unambiguous no-context message guarding against hallucination.

4. **Decision:** LangGraph vs a simple LangChain chain.
Rationale: LangGraph is an explicit and auditable directed graph of control flow.
This conditional advantage that would allow the hallucination guard would be implicit in a chain.
Interview answer: LangGraph allows our agent decision logic to be represented as a directed graph.
containing explicit nodes and edges. The post-retrieval conditional advantage predetermines the performance of the retrieval.
hallucination guard measurable independently and observable in the graph structure.

---

## QA Test Results

*(QA Lead fills this in during Phase 2 of Hour 2)*

| Test | Expected | Actual | Pass / Fail |
|---|---|---|---|
| Normal query | Relevant chunks, source cited | Answer with [SOURCE: RNN or rnn_intermediate.md] | Pass |
| Off-topic query | No context found message | I was unable to find relevant information in the corpus | Pass |
| Duplicate ingestion | Second upload skipped | 0 new chunks 20 duplicates skipped | Pass |
| Empty query | Graceful error, no crash | Chat input disabled when empty | Pass |
| Cross-topic query | Multi-topic retrieval | Retrieved chunks from RNN and LSTM files | Pass |


**Critical failures fixed before Hour 3:**
fixed AgentState dict attribute access fixed in all nodes via state.get() rather than state.attribute<|human|>fixed AgentState dict attribute access fixed in all nodes via state.get() rather than state.attribute
Groq LLM factory stub using API key as environment.

**Known issues not fixed (and why):**
Panel of document viewer is not completely implemented because of the time constraint.
On restarting the application, conversation memory is lost since MemorySaver is in-memory only.

---

## Known Limitations

The memory of conversation is lost when the app restarts.
The similarity threshold of 0.3 had not been empirically calibrated.
Only not full chunk browser is present in document viewer panel.
Headers footers and reference lists could result in noisy PDF chunking.
Streamlit Cloud deployment uses manual corpus reingestion every time it is restarted.

---

## What We Would Do With More Time

Adopt hybrid search with ChromaDB search using vectors and BM25 keyword search.
Include a re-ranking step to achieve a higher retrieval accuracy.
Streamlit UI is not blocked by the ingestion of large PDFs in an asynchronous manner.
Streamlit Cloud persistence with external vectors DB such as Pinecone.
Streaming responses can be added with graph.stream() to provide a better user experience.
Install the entire document viewer and metadata display with chunk browser.

---

## Hour 3 Interview Questions

*(QA Lead fills this in — these are the questions your team
will ask the opposing team during judging)*

**Question 1:What is the vanishing gradient problem and how do LSTMs solve it?

Model answer:According to the context:

The vanishing gradient problem occurs when gradients become exponentially small as they are backpropagated through many time steps, making it difficult for standard RNNs to learn long-range dependencies [SOURCE: RNN | rnn_intermediate.md].

LSTMs solve the vanishing gradient problem through gated mechanisms: the forget gate, input gate, and output gate [SOURCE: LSTM | test_lstm.md]. These gates control information flow through the cell state, allowing the network to maintain relevant information across long sequences [SOURCE: LSTM | test_lstm.md].

In more detail, LSTMs introduce a cell state, a separate memory pathway that runs through the entire sequence, modified by the three learned gates: the forget gate, the input gate, and the output gate [SOURCE: RNN | rnn_intermediate.md]. This gating mechanism allows LSTMs to selectively remember or forget information over long sequences, addressing the vanishing gradient problem.

So, the key point is that LSTMs use gated mechanisms to control information flow through the cell state, enabling them to learn long-range dependencies and solve the vanishing gradient problem.

**Question 2:Why did you choose cosine similarity over dot product for your vector store?

Model answer:According to the context:

The vanishing gradient problem occurs when gradients become exponentially small as they are backpropagated through many time steps, making it difficult for standard RNNs to learn long-range dependencies [SOURCE: RNN | rnn_intermediate.md].

LSTMs solve the vanishing gradient problem through gated mechanisms: the forget gate, input gate, and output gate [SOURCE: LSTM | test_lstm.md]. These gates control information flow through the cell state, allowing the network to maintain relevant information across long sequences [SOURCE: LSTM | test_lstm.md].

In more detail, LSTMs introduce a cell state, a separate memory pathway that runs through the entire sequence, modified by the three learned gates: the forget gate, the input gate, and the output gate [SOURCE: RNN | rnn_intermediate.md]. This gating mechanism allows LSTMs to selectively remember or forget information over long sequences, addressing the vanishing gradient problem.

So, the key point is that LSTMs use gated mechanisms to control information flow through the cell state, enabling them to learn long-range dependencies and solve the vanishing gradient problem.

**Question 3: How does your duplicate detection work and why is content hashing better than filename checking?

Model answer:I don't have any information about cosine similarity, dot products, vector stores, duplicate detection, content hashing, or filename checking in the provided context. These topics don't seem to relate to the context of deep learning architectures (CNN, Autoencoder, Seq2Seq) or RNNs/LSTMs.

Could you provide more context or clarify which specific source this question is related to? I'll be happy to help once I have more information.

---

## Team Retrospective



**What clicked:**
-LangGraph conditional edges made the hallucination guard clean, explicit, and testable
- ChromaDB PersistentClient made local development simple with no external dependencies
- RecursiveCharacterTextSplitter with markdown separators produced good chunk quality
- Streamlit cache_resource kept the embedding model and vector store loaded efficiently

**What confused us:**
-AgentState behaves as a dict not an object so state.attribute access fails silently
- Streamlit cache_resource caches old code requiring full app restart after file changes
- Streamlit Cloud secrets box UI was difficult to edit requiring multiple attempts
- LangGraph node functions must return a dict of state updates not None or the graph stalls

**One thing each team member would study before a real interview:**
- Corpus Architect:Chunk size optimisation and semantic coherence evaluation metrics
- Pipeline Engineer:ChromaDB HNSW index parameters and approximate nearest neighbour theory
- UX Lead:Streamlit async patterns for non-blocking ingestion using threading and st.empty
- Prompt Engineer:Structured output reliability across different LLMs and JSON repair strategies
- QA Lead:Embedding model evaluation using the MTEB benchmark leaderboard
