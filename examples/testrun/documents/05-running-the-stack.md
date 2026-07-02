# Running the stack

A typical vetosh workflow uses three commands, one per component.

First, run `vetosh indexer --config config.yaml`. The indexer reads the
configured sources, parses and chunks each document, embeds the chunks, and
writes the resulting vectors into the configured vector database. In streaming
mode it keeps watching for changes; in static mode it indexes the current
contents once and exits.

Next, run `vetosh server --config config.yaml`. This starts the FastAPI service.
The `POST /retrieve` endpoint embeds a query and returns the top matching chunks.
The `POST /rag` endpoint additionally sends the retrieved context to a language
model and returns a generated answer, but only when an `llm` section is present.

Finally, run `vetosh frontend --config config.yaml` to serve the web chat UI. The
frontend proxies requests to the API at the address given by `frontend.api_url`,
so the browser never talks to the API directly and no CORS configuration is
needed.
