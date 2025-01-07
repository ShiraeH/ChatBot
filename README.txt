TODO
・LocalHostでなく、どこかにサーバー立ててWeb上どこでもアクセス可能に
・PDF/txt以外も処理できるように

----------------------------------------------------
A RUNNNING
----------------------------------------------------
① conda activate Py39
② cd C:\Users\cera0\Documents\Job\Proto241214
③ streamlit run internal_qa.py --server.port 8080
〇 python add_document.py


----------------------------------------------------
B INSTALL
----------------------------------------------------
① Anaconda
② VSCode
③ Microsoft Visual C++ Build Tools
④ 各インストール
    conda create -n Python39 pyhton=3.9
    conda install -c anaconda certifi
　　numpy =1.23.4
　　python-magic
    python-magic-bin
    sympy==1.13.1
    langchain
　　langchain-unstructured
    langchain-openai
    langchain_community
    unstructured
    unstructured-inference
    streamlit
　　python-dotenv
    pinecone-client tiktoken
    pi-heif
    unstructured[image]
    python-magic
    cmake
　　onnx==1.6.1
　　onnxruntime　　
