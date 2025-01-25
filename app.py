import streamlit as st
from  functions.PoetryChain import PoetryChain
from  functions.PoetryComparison import PoetryComparison

gpt2_chain = PoetryChain(model_name="gpt2")
gpt_neo_chain = PoetryChain(model_name="gpt-neo")

comparison = PoetryComparison()

st.title("Poetry Generation - Comparison GPT-2 vs GPT-Neo")
st.write("Enter a topic, and the models will generate poems. Then, the app will compare the poems using BLEU and ROUGE scores.")

user_input = st.text_input("Enter a topic for the poem:")


if st.button("Generate Poetry"):
    if user_input:
        gpt2_poem = gpt2_chain.generate_poem(user_input)
        gpt_neo_poem = gpt_neo_chain.generate_poem(user_input)
        #display
        st.write("### GPT-2 Poem")
        st.write(gpt2_poem)
        
        st.write("### GPT-Neo Poem")
        st.write(gpt_neo_poem)
        
        #b&r score
        comparison_results = comparison.compare_poems(gpt2_poem, gpt_neo_poem)
        
        st.write("### Comparison Scores")
        st.write(f"**BLEU Score**: {comparison_results['BLEU']:.4f}")
        st.write(f"**ROUGE-1 F-Measure**: {comparison_results['ROUGE-1']:.4f}")
        st.write(f"**ROUGE-2 F-Measure**: {comparison_results['ROUGE-2']:.4f}")
        st.write(f"**ROUGE-L F-Measure**: {comparison_results['ROUGE-L']:.4f}")
    else:
        st.write("Please enter a topic for the poem.")
