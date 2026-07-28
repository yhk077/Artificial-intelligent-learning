import os, json, requests, time
os.environ["OPENAI_API_KEY"] = "lm-studio"
os.environ["OPENAI_BASE_URL"] = "http://localhost:1234/v1"
os.environ["MODELSCOPE_CACHE"] = "C:/temp/modelscope"
os.environ["HF_HOME"] = "C:/temp/huggingface"
os.environ["TMPDIR"] = "C:/temp"
print("环境变量配置完成。")

# ============================================================
# 任务零：加载数据集
# ============================================================
print("\n" + "=" * 60)
print("任务零：加载数据集")
print("=" * 60)
from langchain_core.documents import Document
def load_arxiv_corpus(path="data/arxiv_corpus.jsonl"):
    docs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            content = f"Title: {r['title']}\n\nAbstract: {r['abstract']}"
            docs.append(Document(
                page_content=content,
                metadata={
                    "id": r["id"],
                    "title": r["title"],
                    "topic": r["topic"],
                    "authors": ", ".join(r["authors"]),
                    "published": r["published"],
                    "categories": ", ".join(r["categories"]),
                },
            ))
    return docs

corpus = load_arxiv_corpus()
print(f"共加载论文：{len(corpus)}")
print(f"示例 metadata：{corpus[0].metadata}")
print(f"示例正文前200字：{corpus[0].page_content[:200]}")

# ============================================================
# 任务一：构建带元数据的文献 RAG
# ============================================================
print("\n" + "=" * 60)
print("任务一：构建带元数据的文献 RAG")
print("=" * 60)
from modelscope import snapshot_download
print("正在下载嵌入模型 bge-small-zh-v1.5 ...")
embedding_dir = snapshot_download(
    "BAAI/bge-small-zh-v1.5",
    cache_dir="C:/temp/modelscope"
)
print(f"嵌入模型路径：{embedding_dir}")

print("\n正在下载重排模型 bge-reranker-base ...")
reranker_dir = snapshot_download(
    "BAAI/bge-reranker-base",
    cache_dir="C:/temp/modelscope"
)
print(f"重排模型路径：{reranker_dir}")


from langchain_openai import ChatOpenAI
llm = ChatOpenAI(
    model="local-model",
    openai_api_key="lm-studio",
    openai_api_base="http://localhost:1234/v1",
    temperature=0,
)

try:
    test_resp = llm.invoke("请用中文回复：你好")
    print(f"\nLLM 连接测试成功：{test_resp.content[:100]}")
except Exception as e:
    print(f"\n⚠ LLM 连接失败：{e}")
    print("请确保 LM Studio 已启动并加载了 Qwen3-8B 模型！")

print("\n--- 1-1：构建向量索引 ---")

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = HuggingFaceEmbeddings(
    model_name=embedding_dir,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)
vectorstore = InMemoryVectorStore(embedding=embeddings)
vectorstore.add_documents(corpus)
print(f"向量索引构建完成，共 {len(corpus)} 篇论文。")

def paper_rag(query, k=3):
    """
    检索本地论文库，让 LLM 用中文回答。
    上下文带上标题和作者，答案末尾列出引用论文标题。
    """
    docs = vectorstore.similarity_search(query, k=k)
    context_parts = []
    for i, d in enumerate(docs, 1):
        context_parts.append(
            f"[{i}] 标题：{d.metadata['title']}\n"
            f"    作者：{d.metadata['authors']}\n"
            f"    摘要：{d.page_content}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""你是一个科研助手。请严格依据以下检索到的论文内容，用中文回答用户的问题。

如果检索到的论文不足以回答问题，请明确说"根据现有论文库，无法回答该问题"，不要编造任何信息。

## 检索到的论文

{context}

## 用户问题

{query}

## 要求

1. 用中文回答，语言清晰简洁
2. 只依据上面检索到的论文内容，不要使用你自己的知识
3. 答案末尾列出所引用的论文标题（格式：- 标题）"""

    response = llm.invoke(prompt)
    return response.content, docs

print("\n测试 paper_rag：")
answer, docs = paper_rag("有哪些关于探索（exploration）的强化学习方法？", k=3)
print(f"LLM 回答：\n{answer}")
print(f"\n检索到的论文：")
for d in docs:
    print(f"  - [{d.metadata['topic']}] {d.metadata['title']}")

print("\n" + "-" * 40)
print("--- 1-2：分块策略对比 ---")

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=256, chunk_overlap=40,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunked_docs = []
for doc in corpus:
    chunks = splitter.split_documents([doc])
    for chunk in chunks:
        chunk.metadata = doc.metadata.copy()
    chunked_docs.extend(chunks)

print(f"方案A（整条）：{len(corpus)} 个文档")
print(f"方案B（切分，chunk_size=256）：{len(chunked_docs)} 个文档")

vectorstore_chunked = InMemoryVectorStore(embedding=embeddings)
vectorstore_chunked.add_documents(chunked_docs)

test_query = "有哪些关于探索（exploration）的强化学习方法？"
print(f"\n测试问题：{test_query}")

print("\n方案A（整条）Top-3：")
docs_a = vectorstore.similarity_search(test_query, k=3)
for i, d in enumerate(docs_a, 1):
    print(f"  [{i}] {d.metadata['title'][:80]}")

print("\n方案B（切分）Top-3：")
docs_b = vectorstore_chunked.similarity_search(test_query, k=3)
for i, d in enumerate(docs_b, 1):
    print(f"  [{i}] {d.metadata['title'][:80]}")

print("\n" + "-" * 40)
print("--- 1-3：元数据过滤检索 ---")

def paper_rag_filtered(query, topic=None, k=3):
    """只检索指定 topic 的论文"""
    flt = (lambda doc: doc.metadata["topic"] == topic) if topic else None
    docs = vectorstore.similarity_search(query, k=k, filter=flt)

    context_parts = []
    for i, d in enumerate(docs, 1):
        context_parts.append(
            f"[{i}] 标题：{d.metadata['title']}\n"
            f"    作者：{d.metadata['authors']}\n"
            f"    摘要：{d.page_content}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""你是一个科研助手。请严格依据以下检索到的论文内容，用中文回答用户的问题。
如果检索到的论文不足以回答问题，请明确说"根据现有论文库，无法回答该问题"。

## 检索到的论文

{context}

## 用户问题

{query}

## 要求
1. 用中文回答
2. 只依据上面检索到的论文内容
3. 答案末尾列出所引用的论文标题"""

    response = llm.invoke(prompt)
    return response.content, docs

print("\n不加过滤检索：")
answer_all, docs_all = paper_rag_filtered("大模型智能体如何进行工具调用？", topic=None, k=3)
for d in docs_all:
    print(f"  - [{d.metadata['topic']}] {d.metadata['title'][:80]}")

print(f"\n限定 topic='llm-agent' 检索：")
answer_filtered, docs_filtered = paper_rag_filtered("大模型智能体如何进行工具调用？", topic="llm-agent", k=3)
for d in docs_filtered:
    print(f"  - [{d.metadata['topic']}] {d.metadata['title'][:80]}")

# ============================================================
# 任务二：检索质量评测
# ============================================================
print("\n" + "=" * 60)
print("任务二：检索质量评测")
print("=" * 60)

eval_set = [
    {"q": "有哪些用于探索的强化学习方法？",          "gold_topic": "reinforcement-learning"},
    {"q": "Actor-Critic 相关的强化学习研究有哪些？",   "gold_topic": "reinforcement-learning"},
    {"q": "检索增强生成如何减少大模型幻觉？",          "gold_topic": "retrieval-augmented-generation"},
    {"q": "RAG 里怎么做重排序或检索优化？",            "gold_topic": "retrieval-augmented-generation"},
    {"q": "如何评测一个 RAG 系统的效果？",             "gold_topic": "retrieval-augmented-generation"},
    {"q": "大模型智能体如何进行工具调用？",            "gold_topic": "llm-agent"},
    {"q": "多智能体协作方面有哪些工作？",              "gold_topic": "llm-agent"},
    {"q": "让大模型做规划（planning）的方法有哪些？",   "gold_topic": "llm-agent"},
]

def hit_at_k(retrieve_fn, eval_set, k=3):
    hit = 0
    for item in eval_set:
        docs = retrieve_fn(item["q"], k=k)
        topics = [d.metadata["topic"] for d in docs]
        if item["gold_topic"] in topics:
            hit += 1
    return hit / len(eval_set)

from sentence_transformers import CrossEncoder

reranker = CrossEncoder(reranker_dir)

def rerank(query, docs, top_k=3):
    """用 bge-reranker-base 对检索结果重排序"""
    if len(docs) == 0:
        return []
    pairs = [[query, doc.page_content] for doc in docs]
    scores = reranker.predict(pairs)
    scored = list(zip(scores, docs))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:top_k]]

print("\n--- 2-3：三种检索方案 Hit@3 对比 ---")

def retrieve_baseline(q, k=3):
    return vectorstore.similarity_search(q, k=k)

hit_a = hit_at_k(retrieve_baseline, eval_set, k=3)
print(f"方案A（纯向量检索 k=3）：Hit@3 = {hit_a:.2%}")

def retrieve_with_rerank(q, k=3):
    docs = vectorstore.similarity_search(q, k=6)
    return rerank(q, docs, top_k=k)

hit_b = hit_at_k(retrieve_with_rerank, eval_set, k=3)
print(f"方案B（k=6召回 → 重排Top-3）：Hit@3 = {hit_b:.2%}")

print("\n--- 2-4：改进方案（查询翻译 + 重排序）---")

def translate_to_english(query):
    """让 LLM 把中文问题翻译成英文，用于跨语言检索"""
    prompt = f"Translate the following Chinese question into English. Only output the English translation, nothing else.\n\nChinese: {query}\n\nEnglish:"
    response = llm.invoke(prompt)
    return response.content.strip()

def retrieve_improved(q, k=3):
    en_q = translate_to_english(q)
    docs = vectorstore.similarity_search(en_q, k=10)
    return rerank(en_q, docs, top_k=k)

hit_c = hit_at_k(retrieve_improved, eval_set, k=3)
print(f"方案C（查询翻译+扩大召回+重排）：Hit@3 = {hit_c:.2%}")

print("\n逐题命中明细（方案C）：")
for item in eval_set:
    en_q = translate_to_english(item["q"])
    docs = retrieve_improved(item["q"], k=3)
    topics = [d.metadata["topic"] for d in docs]
    hit = "✓" if item["gold_topic"] in topics else "✗"
    print(f"  {hit} {item['q'][:40]}... → 检索到 topics: {topics}")

print("\n" + "-" * 50)
print("| 检索方案 | Hit@3 |")
print("|---|---|")
print(f"| A. 纯向量检索（k=3） | {hit_a:.2%} |")
print(f"| B. 向量召回 k=6 → 重排取 Top-3 | {hit_b:.2%} |")
print(f"| C. 查询翻译+扩大召回+重排 | {hit_c:.2%} |")

# ============================================================
# 任务三：封装真实世界工具
# ============================================================
print("\n" + "=" * 60)
print("任务三：封装真实世界工具")
print("=" * 60)

import xml.etree.ElementTree as ET
from langchain_core.tools import tool

# 工具 A 
@tool
def search_local_papers(query: str) -> str:
    """检索本地已收录的强化学习/RAG/智能体论文库（共90篇）。
    当用户问'本地库/已收录论文'或询问这三个领域的具体方法、综述时使用。
    输入中英文均可，返回论文标题和作者。"""
    try:
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "本地论文库中没有找到相关论文。"
        lines = []
        for i, d in enumerate(docs, 1):
            lines.append(f"[{i}] {d.metadata['title']}\n"
                         f"    作者：{d.metadata['authors']}\n"
                         f"    主题：{d.metadata['topic']}")
        return "\n".join(lines)
    except Exception as e:
        return f"本地文献检索失败：{e}"

# 工具 B
@tool
def search_arxiv(query: str) -> str:
    """按关键词在 arXiv 上检索最新论文。当用户问'最近/最新有哪些关于X的论文'、
    或本地知识库里可能没有的新工作时使用。输入英文关键词效果最好。"""
    try:
        r = requests.get("http://export.arxiv.org/api/query",
            params={"search_query": f"all:{query}", "start": 0, "max_results": 5,
                    "sortBy": "submittedDate", "sortOrder": "descending"}, timeout=30)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        root = ET.fromstring(r.text)
        out = []
        for e in root.findall("a:entry", ns):
            title = " ".join(e.find("a:title", ns).text.split())
            pub = e.find("a:published", ns).text[:10]
            out.append(f"[{pub}] {title}")
        return "\n".join(out) if out else "没有检索到相关论文。"
    except Exception as e:
        return f"arXiv 检索失败：{e}"

# 工具 C
@tool
def paper_citations(title: str) -> str:
    """查询一篇论文的被引用次数、发表年份和作者。输入论文标题（英文）。
    当用户问某篇论文'被引多少次''影响力如何''谁写的''哪年发的'时使用。"""
    try:
        r = requests.get("https://api.crossref.org/works",
            params={"query.bibliographic": title, "rows": 1}, timeout=30)
        items = r.json().get("message", {}).get("items", [])
        if not items:
            return "Crossref 未找到该论文。"
        it = items[0]
        t = (it.get("title") or ["(无标题)"])[0]
        year = (it.get("issued", {}).get("date-parts", [[None]])[0][0])
        cited = it.get("is-referenced-by-count", "未知")
        authors = ", ".join(f"{a.get('given','')} {a.get('family','')}".strip()
                            for a in it.get("author", [])[:3])
        return f"标题：{t}\n年份：{year}\n被引次数：{cited}\n作者：{authors}"
    except Exception as e:
        return f"Crossref 查询失败：{e}"

# 工具 D
@tool
def wiki_lookup(term: str) -> str:
    """查询一个术语或概念的定义/简介（中文维基百科）。当用户问'什么是X''解释一下X'
    且本地论文库不足以解释这个通用概念时使用。"""
    try:
        r = requests.get(f"https://zh.wikipedia.org/api/rest_v1/page/summary/{term}",
                         timeout=20, headers={"User-Agent": "edu-lab/1.0"})
        if r.status_code == 200:
            extract = r.json().get("extract", "")
            return extract[:400] if extract else "维基百科没有该词条的摘要。"
        return f"维基百科未找到词条「{term}」（状态码 {r.status_code}）。"
    except Exception as e:
        return f"维基百科查询失败：{e}"

# 工具 E
@tool
def get_weather(city: str) -> str:
    """查询某个城市的实时天气（温度、湿度、风速）。输入城市名（中英文均可）。"""
    try:
        g = requests.get("https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "zh"}, timeout=20).json()
        if not g.get("results"):
            return f"找不到城市「{city}」。"
        loc = g["results"][0]
        w = requests.get("https://api.open-meteo.com/v1/forecast",
            params={"latitude": loc["latitude"], "longitude": loc["longitude"],
                    "current": "temperature_2m,relative_humidity_2m,wind_speed_10m"},
            timeout=20).json()["current"]
        return (f"{loc['name']} 当前天气：温度 {w['temperature_2m']}°C，"
                f"湿度 {w['relative_humidity_2m']}%，风速 {w['wind_speed_10m']} m/s")
    except Exception as e:
        return f"天气查询失败：{e}"

# 工具 F
@tool
def currency_convert(query: str) -> str:
    """货币汇率换算。输入格式：'金额 源币种 目标币种'，例如 '250 USD CNY'。
    币种用三位字母代码（USD/CNY/EUR/JPY 等）。"""
    try:
        amount, base, target = query.split()
        r = requests.get("https://api.frankfurter.dev/v1/latest",
            params={"base": base.upper(), "symbols": target.upper()}, timeout=20).json()
        rate = r["rates"][target.upper()]
        return f"{amount} {base.upper()} = {float(amount) * rate:.2f} {target.upper()}（汇率 {rate}，{r['date']}）"
    except Exception as e:
        return f"汇率换算失败：{e}"

# 工具 G
@tool
def calculator(expression: str) -> str:
    """计算数学表达式。输入数学表达式如 '(128 + 256) * 3'，返回计算结果。"""
    try:
        allowed = set("0123456789+-*/.() ")
        if not all(c in allowed for c in expression):
            return "表达式包含不允许的字符。"
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算出错：{e}"

all_tools = [
    search_local_papers,
    search_arxiv,
    paper_citations,
    wiki_lookup,
    get_weather,
    currency_convert,
    calculator,
]

print("已封装的工具：")
for t in all_tools:
    print(f"  - {t.name}: {t.description[:80]}...")

print("\n--- 工具验证 ---")

print("\n1. search_local_papers('强化学习探索方法'):")
print(search_local_papers.invoke("强化学习探索方法")[:300])

print("\n2. search_arxiv('diffusion policy'):")
print(search_arxiv.invoke("diffusion policy")[:300])

print("\n3. paper_citations('Attention is all you need'):")
print(paper_citations.invoke("Attention is all you need")[:300])

print("\n4. wiki_lookup('检索增强生成'):")
print(wiki_lookup.invoke("检索增强生成")[:300])

print("\n5. get_weather('Singapore'):")
print(get_weather.invoke("Singapore")[:300])

print("\n6. currency_convert('250 USD CNY'):")
print(currency_convert.invoke("250 USD CNY")[:300])

print("\n7. calculator('(128 + 256) * 3'):")
print(calculator.invoke("(128 + 256) * 3")[:300])

# ============================================================
# 任务四：ReAct 多工具智能体 + 记忆
# ============================================================
print("\n" + "=" * 60)
print("任务四：ReAct 多工具智能体 + 记忆")
print("=" * 60)

system_prompt = """你是一个科研情报智能体，可以帮助用户检索论文、查询学术信息、查天气和计算。

你有以下工具可用：

- search_local_papers: 检索**本地已收录**的强化学习/RAG/智能体论文库（共90篇）。当用户问这些领域的已有论文、方法综述时使用。
- search_arxiv: 在 arXiv 上**实时检索最新**论文。当用户问"最近/最新"的论文、或本地库覆盖不到的新领域时使用。
- paper_citations: 查询某篇论文的**被引次数、作者、发表年份**。当用户问"被引多少次""谁写的""影响力"时使用。
- wiki_lookup: 查询术语/概念的**定义解释**（中文维基百科）。当用户问"什么是X"且本地论文库不足以解释时使用。
- get_weather: 查询城市**实时天气**（温度、湿度、风速）。
- currency_convert: 货币汇率**换算**。输入格式："金额 源币种 目标币种"，如 "250 USD CNY"。
- calculator: 计算数学表达式。

重要规则：
1. 区分 search_local_papers 和 search_arxiv：
   - "本地/已收录/库里有" → search_local_papers
   - "最新/最近/新的" → search_arxiv
2. 区分 paper_citations 和 search_arxiv：
   - "被引/引用次数/谁写的/影响力" → paper_citations
   - "找论文/搜索" → search_arxiv 或 search_local_papers
3. 每次只调用一个工具。
4. 用中文回答用户，最终答案要完整清晰。

请严格按照以下格式回复：

Thought: [你的思考过程]
Action: [工具名称]
Action Input: [输入参数]

当获得 Observation 后，继续思考直到得出最终答案：

Thought: 我现在知道最终答案了
Final Answer: [用中文给出最终答案]"""

def run_agent(user_query, tools, llm, system_prompt_text=None, history=None):
    """ReAct 循环：Thought → Action → Observation → ... → Final Answer"""
    if system_prompt_text is None:
        system_prompt_text = system_prompt

    tools_map = {t.name: t for t in tools}

    messages = [{"role": "system", "content": system_prompt_text}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_query})

    max_turns = 10
    for turn in range(max_turns):
        response = llm.invoke(messages)
        content = response.content

        print(f"\n{'='*50}")
        print(f"--- Turn {turn + 1} ---")
        print(content)

        if "Final Answer:" in content:
            final = content.split("Final Answer:")[-1].strip()
            print(f"\n{'='*50}")
            print(f"最终答案：{final}")
            return final

        if "Action:" in content and "Action Input:" in content:
            try:
                action_part = content.split("Action:")[-1].split("Action Input:")[0].strip()
                action_input = content.split("Action Input:")[-1].strip()

                print(f"\n>>> 执行工具：{action_part}({action_input})")

                tool = tools_map.get(action_part.strip())
                if tool:
                    result = tool.invoke(action_input.strip())
                    observation = str(result)
                else:
                    observation = f"工具 '{action_part}' 不存在。可用工具：{list(tools_map.keys())}"

                print(f">>> Observation: {observation[:500]}{'...' if len(observation) > 500 else ''}")

                messages.append({"role": "assistant", "content": content})
                messages.append({"role": "user", "content": f"Observation: {observation}"})

            except Exception as e:
                print(f">>> 解析错误：{e}")
                messages.append({"role": "user", "content": f"解析失败：{e}。请严格使用 Thought/Action/Action Input 格式。"})
        else:
            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": "请继续。如果需要使用工具，请给出 Action 和 Action Input。如果已经得到答案，请给出 Final Answer。"})

    return "Agent 达到最大轮次限制。"

print("\n--- 4-2：五题测试 ---")

test_questions = [
    "强化学习里做探索（exploration）的方法，本地库里收录了哪些？",
    "最近有哪些关于 diffusion policy 的新论文？",
    '"Attention is all you need" 这篇论文被引用了多少次？',
    "帮我算一下 (128 + 256) * 3 等于几",
    "什么是检索增强生成（RAG）？",
]

for i, q in enumerate(test_questions, 1):
    print(f"\n{'#'*60}")
    print(f"# 题目 {i}：{q}")
    print(f"{'#'*60}")
    try:
        result = run_agent(q, all_tools, llm)
    except Exception as e:
        print(f"题目 {i} 执行出错：{e}")
print("\n" + "=" * 60)
print("--- 4-3：复合指令测试 ---")
print("=" * 60)

compound_query = (
    "我下个月想去新加坡参加一个学术会议。帮我："
    "①查一下新加坡现在天气怎么样；"
    "②会议注册费 250 美元，折合多少人民币；"
    "③顺便在 arXiv 上找几篇最近关于 llm agent 的论文。"
)

print(f"\n复合指令：{compound_query}")
try:
    result = run_agent(compound_query, all_tools, llm)
except Exception as e:
    print(f"复合指令执行出错：{e}")

print("\n" + "=" * 60)
print("--- 4-4：多轮对话记忆 ---")
print("=" * 60)

print("\n>>> 第 1 轮：帮我在 arXiv 上找找 reinforcement learning 的最新论文。")
round1_query = "帮我在 arXiv 上找找 reinforcement learning 的最新论文。"
memory = []  

try:
    answer1 = run_agent(round1_query, all_tools, llm, history=memory)
    memory.append({"role": "user", "content": round1_query})
    memory.append({"role": "assistant", "content": answer1})
except Exception as e:
    print(f"第1轮出错：{e}")

print("\n>>> 第 2 轮：其中第一篇，帮我查查它被引用了多少次。")
round2_query = "其中第一篇，帮我查查它被引用了多少次。"

try:
    answer2 = run_agent(round2_query, all_tools, llm, history=memory)
except Exception as e:
    print(f"第2轮出错：{e}")

print("\n" + "=" * 60)
print("全部任务代码执行完毕。")
print("=" * 60)
