for i in $(seq 1 5);
do
    mkdir -p sim{$i}
    cd sim{$i}
    mkdir -p agent_out
    mkdir -p gpt_out
    mkdir -p gpt_out
    mkdir -p search_out
    cd ..
done

python pipeline.py
python search_only.py
python pdf_only.py
python gpt_pipeline.py