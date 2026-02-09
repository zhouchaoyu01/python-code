import streamlit as st
import requests
import os
import uuid

# --- 配置区 ---
# 指向你刚才启动的 FastAPI 地址
BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="AI 企业知识库助手",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 企业级 RAG 问答助手")
st.markdown("---")
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
# --- 侧边栏：文档上传 ---
with st.sidebar:
    st.header("知识库管理")
    uploaded_file = st.file_uploader("上传文档 (PDF, DOCX, TXT)", type=["pdf", "docx", "txt"])
    
    if st.button("🚀 开始处理文档"):
        if uploaded_file is not None:
            with st.spinner("正在解析并入库，请稍候..."):
                try:
                    # 将 Streamlit 的文件对象发送给 FastAPI
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                    response = requests.post(f"{BASE_URL}/upload", files=files)
                    
                    if response.status_code == 200:
                        st.success(f"✅ {uploaded_file.name} 处理成功！")
                        st.json(response.json())
                    else:
                        st.error(f"❌ 上传失败: {response.text}")
                except Exception as e:
                    st.error(f"连接后端失败: {e}")
        else:
            st.warning("请先选择一个文件")

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.subheader("已加载文档")
    files = requests.get(f"{BASE_URL}/files").json()
    for f in files:
        col1, col2 = st.columns([0.8, 0.2])
        col1.caption(f"📄 {f['file_name']}")
        if col2.button("🗑️", key=f['file_hash']):
            requests.delete(f"{BASE_URL}/files/{f['file_hash']}")
            st.rerun()
# --- 主界面：聊天窗口 ---

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 接收用户输入
if prompt := st.chat_input("请输入您关于文档的问题..."):
    # 1. 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端接口获取回答
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            try:
                response = requests.post(
                    f"{BASE_URL}/chat/v2",
                    json={"query": prompt, "session_id": st.session_state.session_id},
                    timeout=60 # RAG 有时检索较慢，设置较长超时
                )
                
                if response.status_code == 200:
                    answer = response.json().get("answer", "未获取到回答")
                    sources = response.json().get("sources", [])
                    st.markdown(answer)
                    # 如果有来源，展示来源标签
                    if sources:
                        with st.expander("查看参考来源"):
                            for s in sources:
                                # 这里的 s['index'] 完美对应回答里的 [1], [2]
                                st.write(f"**[{s['index']}] {s['file_name']}**")
                                st.caption(f"内容摘要: {s['content']}")
                    else:
                        pass
                    # 保存到历史
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"后端返回错误: {response.status_code}")
            except Exception as e:
                st.error(f"请求失败: {e}")