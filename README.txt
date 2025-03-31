TODO
・LocalHostでなく、どこかにサーバー立ててWeb上どこでもアクセス可能に
・PDF/txt以外も処理できるように

----------------------------------------------------
A RUNNNING(local)
----------------------------------------------------
① conda activate YOUR-ENVIRONMETN
② cd YOUR-PYTHON-FILE-PATH
③ streamlit run internal_qa.py --server.port 8080
〇 python add_document.py


----------------------------------------------------
B INSTALL
----------------------------------------------------
① Anaconda
② VSCode
③ Microsoft Visual C++ Build Tools
④ tess eractのインストール（https://github.com/UB-Mannheim/tesseract/wiki/）
⑤ 各インストール
    conda create -n Python39 pyhton=3.9
    conda install -c anaconda certifi
　　その他ライブラリはrequirements.txtを参照
