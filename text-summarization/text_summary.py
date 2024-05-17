import torch
import gradio as gr


# Use a pipeline as a high-level helper
from transformers import pipeline

model_path = ("../Models/models--sshleifer--distilbart-cnn-12-6/snapshots/"
              "a4f8f3ea906ed274767e9906dbaede7531d660ff")
text_summary = pipeline("summarization", model=model_path,
                torch_dtype=torch.bfloat16)


text="""
In probability theory and statistics, Bayes' theorem (alternatively Bayes' 
law or Bayes' rule), named after Thomas Bayes, describes the probability of 
an event, based on prior knowledge of conditions that might be related to 
the event.[1] For example, if the risk of developing health problems is 
known to increase with age, Bayes' theorem allows the risk to an individual 
of a known age to be assessed more accurately by conditioning it relative 
to their age, rather than assuming that the individual is typical of the 
population as a whole.

One of the many applications of Bayes' theorem is Bayesian inference, 
a particular approach to statistical inference. When applied, the 
probabilities involved in the theorem may have different probability 
interpretations. With Bayesian probability interpretation, the theorem 
expresses how a degree of belief, expressed as a probability, should 
rationally change to account for the availability of related evidence. 
Bayesian inference is fundamental to Bayesian statistics. It has been 
considered to be "to the theory of probability what Pythagoras's theorem 
is to geometry."[2]

Based on Bayes law both the prevalence of a disease in a given population 
and the error rate of an infectious disease test have to be taken into account 
to evaluate the meaning of a positive test result correctly and avoid the 
base-rate fallacy.
"""

# returns a list
# print(text_summary(text))

def summary(input):
    output = text_summary(input)
    return output[0]['summary_text']

gr.close_all()

# demo = gr.Interface(fn=summary, inputs="text", outputs="text")
demo = gr.Interface(fn=summary,
                    inputs=[gr.Textbox(label="Input text to summarize",
                                       lines=6)],
                    outputs=[gr.Textbox(label="Summarized text",
                                        lines=4)],
                    title="@KitwanaAkil Project 1: Text Summarizer",
                    description="This application will be used to summarize text.")
demo.launch()