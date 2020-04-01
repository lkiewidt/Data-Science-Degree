# P-04: Data Science blog post

## Introduction
The Data Science blog post is part of Udacity's Data Scientist Nanodegree. Within the project, data from [Nature's 2017 Graduate Survey](https://www.nature.com/nature/journal/v550/n7677/full/nj7677-549a.html) was analyzed to obtain insights into the motivation of students to pursue a PhD, the skills they learn during their PhD, and their satisfaction with their PhD program.

## Approach
First, the data was transformed into csv-format and read into a Pandas data frame. After, exploration useful features were selected. Most of the features in the original dataset contained text-based values. Therefore, ordinal features were numerically encoded to allow statistical analysis of the data. Finally, distributions for the motivation of PhD students, the skills they learn during their PhD, and their satisfaction with their PhD program were visualized and analyzed.

## Python Libraries
The following Python libraries are use in the project:
* python = 3.7.3
* numpy = 1.16.4
* pandas = 0.24.2
* matplotlib = 3.1.0
* seaborn = 0.9.0

## Key Findings
* proper selection and encoding of the features into categorical and ordinal features was crucial to analyze the data as most of the ordinal features were encoded as text
* most PhD students pursue a PhD to get an academic position (44%) or to continue to work in research (35%)
    * in chemistry and engineering a non-academic career was mentioned twice as often as the average
* during their PhD are well trained in designing experiments, collecting and analyzing data, and communication of their results to experts through presentations and scientific articles
* skills that are not well trained during a PhD a managing budgets and creating business plans
* overall, 75% of PhD students are at least somewhat satisfied with their PhD program
* PhD students who chose to pursue a PhD due to job requirements, for example a non-academic position, or who wanted to live abroad are significantly less satisfied with their PhD, in particular if they take longer to graduate than the average duration of their respective PhD program
    * for students who mention an academic career or continuation of research as their main motivation, the decrease in satisfaction cannot be observed, even if they are in a delayed stage
* no significant differences in satisfaction between students with different motivation or different fields were not observed
* therefore, the satisfaction score cannot be predicted based on features that are known before the students start their PhD (e.g. field and motivation)
