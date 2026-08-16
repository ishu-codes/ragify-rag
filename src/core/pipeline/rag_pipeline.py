class RAGPipeline:
    def __init__(self, retriever, reranker=None, generator=None):
        self.retriever = retriever
        self.reranker = reranker
        self.generator = generator

    def retrieve(self, query, k=5):
        chunks = self.retriever.retrieve(query, k=20)

        if self.reranker:
            chunks = self.reranker.rerank(query, chunks)

        return chunks[:k]

    def generate(self, query, chunks):
        context = "\n\n".join([c.text for c in chunks])
        return self.generator.generate(query, context)

    def run(self, query, k=5):
        chunks = self.retrieve(query, k)
        answer = self.generate(query, chunks) if self.generator else None
        return chunks, answer
