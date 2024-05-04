import ollama
from datetime import datetime
from chromadb import PersistentClient, EmbeddingFunction, Documents, Embeddings
import json


class MyEmbeddingFunction(EmbeddingFunction):
    """Embedding function for Ollama embeddings."""

    def vectorize(self, text: str) -> list[float]:
        """Vectorize the text using Ollama embeddings.

        :param text: Text to be vectorized.
        :return: Vector representation of the text.
        """
        result = ollama.embeddings(
            model="all-minilm:l6",
            prompt=text,
        )
        return result["embedding"]

    def __call__(self, input: Documents) -> Embeddings:
        """Embed the input documents.

        :param input: Input documents.
        :return: Embeddings of the input documents.
        """
        return [self.vectorize(doc) for doc in input]


class KnowledgeStore(object):
    """KnowledgeStore class to store and retrieve documents."""

    def __init__(self, name: str) -> None:
        """KnowledgeStoreの初期化を行います。

        :param name: データベースの名前
        """
        client = PersistentClient(
            path="data/chromadb",
        )
        self.collection = client.get_or_create_collection(
            name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=MyEmbeddingFunction(),
        )

    def add(self, id: str, text: str) -> None:
        """新しい文書を追加します。

        :param id: 文書のID
        :param text: 文書のテキスト
        """
        self.collection.upsert(
            documents=[text],
            metadatas=[
                {
                    "timestamp": datetime.now().isoformat(),
                }
            ],
            ids=[id],
        )

    def delete(self, id: str) -> None:
        """文書を削除します。

        :param id: 文書のID
        """
        self.collection.delete(ids=[id])

    def query(self, text: str, k: int = 1):
        return self.collection.query(
            query_texts=[text],
            n_results=1,
        )


def retrieve_documents(text: str) -> str:
    """与えられたテキストについて、検索することができます。"""
    agent = KnowledgeStore(name="conversations")
    result = agent.query(text)
    return json.dumps(
        {
            "context": result["documents"][0],
        },
        ensure_ascii=False,
    )


if __name__ == "__main__":
    stores = KnowledgeStore(name="conversations")
    stores.add(
        id="test202405041256",
        text="""
管轄する領域は東京都区部（東京23区）、多摩地域（26市[注釈 4]と西多摩郡3町1村）および東京都島嶼部（大島、三宅、八丈、小笠原）の4支庁（2町7村）からなっている。沖ノ鳥島、南鳥島を含む小笠原諸島を含むため、日本最南端および最東端に位置する都道府県でもある。東京都に対して公式に用いられる英語名称は"Tokyo Metropolis"である[6]。Metropolis自体に法令上の定義は存在しないが、一般には「（周辺都市に対する）中核都市・主要都市」「母都市(mother city)」「首都」の語義で使用される[7]。
# 人口は14,133,086人（2024年4月1日現在）。これは日本の都道府県の中では人口が最も多く、日本の人口のおよそ11%を占めている。人口密度も都道府県の中で最も高い。東京都を中心とする東京都市圏は人口3700万人を超える世界最大の都市圏である。日本の人口の約3割が集中し、ポーランドやモロッコ、カナダなどの国の総人口に匹敵する。
                          """,
    )
    stores.add(
        id="test202405041355",
        text="""
『枕草子』（まくらのそうし）とは、平安時代中期に中宮定子に仕えた女房、清少納言により執筆されたと伝わる随筆。ただし本来は、助詞の「の」を入れずに「まくらそうし」と呼ばれたという。
執筆時期は正確には判明していないが、長保3年（1001年）にはほぼ完成したとされている。「枕草紙」「枕冊子」「枕双紙」とも表記され、古くは『清少納言記』『清少納言抄』などとも称された。また日本三大随筆の一つである。
                          """,
    )

    from skill_agent import SkillAgent

    skill_agent = SkillAgent([retrieve_documents])
    for buffer in skill_agent.use("枕草子の作者は？"):
        print(buffer)
