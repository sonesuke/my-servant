import json

import ollama
from message_log import MessageLog, Role
from prompts import SUMMARIZE_PROMPT, SUMMARY_SCHEMA
from jsonschema import validate


class SummarizeAgent:
    """Summarize agent that uses Ollama API to summarize the chat log."""

    def summarize(self, message_log: MessageLog, retries: int = 3) -> str:
        """Summarize the chat log using Ollama API.

        :param message_log: Log of messages exchanged between the user and the assistant.
        :return: Summary of the chat log.
        """
        messages = [
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {
                "role": "user",
                "content": json.dumps(message_log.get_messages(), ensure_ascii=False),
            },
        ]

        for i in range(retries):
            try:
                response = ollama.chat(
                    model="suzume-mul",
                    messages=messages,
                    stream=False,
                )
                response_json = json.loads(response["message"]["content"])
                validate(instance=response_json, schema=SUMMARY_SCHEMA)
                return response_json
            except Exception:
                pass
        raise Exception("max retries exceeded.")


if __name__ == "__main__":
    summarize_agent = SummarizeAgent()
    message_log = MessageLog()
    message_log.push_new_message(Role.USER, "こんにちは")
    message_log.push_new_message(Role.ASSISTANT, "こんにちは")
    message_log.push_new_message(Role.USER, "日本の首都は？")
    message_log.push_new_message(Role.ASSISTANT, "京都です。")
    message_log.push_new_message(Role.USER, "違います。東京ですね。")
    message_log.push_new_message(Role.ASSISTANT, "わかりました。日本の首都は東京です。")
    message_log.push_new_message(Role.USER, "人口は？")
    message_log.push_new_message(Role.ASSISTANT, "約1000万人です。")
    message_log.push_new_message(Role.USER, "東京都はどの地方にあるんだっけ？")
    message_log.push_new_message(Role.ASSISTANT, "関東地方です。")
    message_log.push_new_message(Role.USER, "東京について調べて")
    message_log.push_new_message(
        Role.ASSISTANT,
        """
管轄する領域は東京都区部（東京23区）、多摩地域（26市[注釈 4]と西多摩郡3町1村）および東京都島嶼部（大島、三宅、八丈、小笠原）の4支庁（2町7村）からなっている。沖ノ鳥島、南鳥島を含む小笠原諸島を含むため、日本最南端および最東端に位置する都道府県でもある。東京都に対して公式に用いられる英語名称は"Tokyo Metropolis"である[6]。Metropolis自体に法令上の定義は存在しないが、一般には「（周辺都市に対する）中核都市・主要都市」「母都市(mother city)」「首都」の語義で使用される[7]。
人口は14,133,086人（2024年4月1日現在）。これは日本の都道府県の中では人口が最も多く、日本の人口のおよそ11%を占めている。人口密度も都道府県の中で最も高い。東京都を中心とする東京都市圏は人口3700万人を超える世界最大の都市圏である。日本の人口の約3割が集中し、ポーランドやモロッコ、カナダなどの国の総人口に匹敵する。""",
    )
    response = summarize_agent.summarize(message_log)
    print(response)
