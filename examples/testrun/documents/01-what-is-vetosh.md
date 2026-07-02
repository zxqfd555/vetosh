# What is vetosh?

vetosh is a universal, no-code Retrieval-Augmented Generation (RAG) server. You
point it at your documents, choose a vector database and an embedder in a single
YAML file, and it stands up a live RAG pipeline without writing any code.

The project has three decoupled components that can run on separate workers:

- The **indexer** watches your sources, parses and chunks documents, embeds the
  chunks, and keeps the vector database in sync in real time.
- The **server** is an async FastAPI service that embeds an incoming query and
  retrieves the most relevant chunks, optionally answering them with an LLM.
- The **frontend** is a small web chat UI that proxies requests to the server.

Because the three are independent processes that share only the vector database,
you can scale each of them on its own and place them in different availability
zones.
