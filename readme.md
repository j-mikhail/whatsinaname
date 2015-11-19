# What's in a name?
##### A machine learning exercise on multiclass classification of natural language
by Jonathan Mikhail

### Background

Ask fans to describe the plot of of their favorite movie, and they can usually do so quickly and comprehensively. But are movie plots simply a random description of a story, divorced from its genre, or is there a pattern which correlates these features? I would like to perform analysis to determine whether such a correlation exists, whether a classification model can be built from it, and whether this model can be used to predict a movie’s genre based only on its plot.

### Literature

The following papers may be of interest:
* [*RTextTools: A Supervised Learning Package for Text Classification*](https://journal.r-project.org/archive/2013-1/collingwood-jurka-boydstun-etal.pdf) by Timothy P. Jurka, Loren Collingwood, Amber E. Boydstun, Emiliano Grossman, and Wouter van Atteveldt
* [*A Practical Guide to Support Vector Classification*](http://www.csie.ntu.edu.tw/~cjlin/papers/guide/guide.pdf) by Chih-Wei Hsu, Chih-Chung Chang, and Chih-Jen Lin
* [*Supervised Term Weighting for Automated Text Categorization*](http://www.nmis.isti.cnr.it/debole/articoli/SAC03b.pdf) by Franca Debole, Fabrizio Sebastiani
* [*Weighted support vector machine for classification with uneven training class sizes*](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1527706&url=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D1527706) by Yi-Min Huang, Shu-xin Du
* [*The Entire Regularization Path for the Support Vector Machine*](http://jmlr.csail.mit.edu/papers/volume5/hastie04a/hastie04a.pdf) by Trevor Hastie, Saharon Rosset, Robert Tibshirani, Ji Zhu
* [*Obtaining calibrated probability estimates from decision trees and naive Bayesian classifiers*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.3039&rep=rep1&type=pdf) by Bianca Zadrozny, Charles Elkan
* [*A Tutorial on Support Vector Machines for Pattern Recognition*](http://research.microsoft.com/pubs/67119/svmtutorial.pdf) by Christopher J.C. Burges
* [*Scorecard construction with unbalanced class sizes*](http://fic.wharton.upenn.edu/fic/handpaper.pdf) by David J. Hand, Veronica Vinciotti
* [*Visualisation of multi-class ROC surfaces*](http://users.dsic.upv.es/~flip/ROCML2005/papers/fieldsend2CRC.pdf) by Jonathan E. Fieldsend, Richard M. Everson
* [*Approximating the multiclass ROC by pairwise analysis*](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.108.3250&rep=rep1&type=pdf) by Thomas C.W. Landgrebe, Robert P.W. Duin

### Data Source

To perform my analysis, I’ve used the September 11, 2015 release of the Internet Movie Database (IMDB), the latest version of which is available for download [here](http://www.imdb.com/interfaces). The data is in compressed plain text, column aligned or tab delineated and prefaced with one or more introductory sections, with each file containing one major movie attribute.

### Data Cleaning

Several of IMDB's data sets use differing data formats, which makes cleaning them a bit of a challenge. In each case, the header and footer paragraphs can be removed manually, however formatting the fields requires more work. *ratings.list*, for example, is column-delineated as follows:
```text
      2....24..2       5   6.2  Back to the Fuchsia (2008)
      0000001222  636253   8.5  Back to the Future (1985)
      6100.0...0      24   2.0  Back to the Future (1989) (VG)
```
I used Excel's data import wizard, which features an option to import column-delineated data, opting to ignore the rating distribution data on the left. Once imported, Excel can save the data as a tab-delimited file.

*plot.list* also has a strange format, which looks something like this:
```text
-------------------------------------------------------------------------------
MV: Back to the 80's: An Interview with Donald P. Borchers (2011) (V)

PL: Prolific producer Donald P. Borchers sits down to talk about his amazing
PL: career and his work on the classic horror movie Vamp. Produced for the
PL: Arrow Films DVD and Blu Ray release which has been available in the UK
PL: since February 21st 2011.

BY: Calum Waddell
```
Each paragraph is split onto multiple lines, each prefixed with a code. I used [Notepad++](https://notepad-plus-plus.org/) for this problem, using Regex-based search and replace functions to remove dotted lines and authors, eliminate the prefixes, and put the data on a single line. For example:

* Replace ***^(BY:)(.*)\n** with blank.
* Replace ***^(-*)\n** with blank.
* Replace **\nPL:** with blank.

Note that if your computer lacks the memory to open the entire file, you can split it first using a tool like [HJSplit](http://www.hjsplit.org/) and work on each file chunk separately, joining them when you're ready to import.

Finally, the remaining files were all tab-delimited, but inconsistently so, with fields separated with one or more tabs depending on text length. For example:
```text
Back to Life (2012)					English
Back to Life (2015)					Italian
Back to Life After 2,000 Years (1910)			French
```
This is a quick fix in Notepad++ by simply searching for **(\t+)** and replacing it with **(\t)**.

### Data Importing

Once we have our documents in a clean, tab-delimited format, importing them into R is pretty easy:
```r
data_ratings <- read.delim("./GitHub/whatsinaname/data/cleaned/ratings.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("rcount","rating","title"))
data_languages <- read.delim("./GitHub/whatsinaname/data/cleaned/language.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","language","variant"))
data_countries <- read.delim("./GitHub/whatsinaname/data/cleaned/countries.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","country"))
data_running_times <- read.delim("./GitHub/whatsinaname/data/cleaned/running-times.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","time","note"))
data_genres <- read.delim("./GitHub/whatsinaname/data/cleaned/genres.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","genre"))
data_plot <- read.delim("./GitHub/whatsinaname/data/cleaned/plot.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","plot"))
```

### Data Structuring

##### Merging
Since R creates a separate data frame for each imported file, we can now merge these into a unified data frame by matching on title:
```r
data_all <- data.frame(data_ratings["title"], data_ratings["rating"], data_ratings["rcount"])
data_all <- merge(data_all, data_running_times, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_languages, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_countries, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_genres, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_plot, by.x = "title", by.y = "title")
```
##### Filtering
Our merges have produced a sizeable data frame, however not all the data contained is pertinent. For simplicity's sake, we would like to restrict our analysis to English language, American-produced theatrical films, with at least 100 ratings.

Another important note is that movies are categorized into several different genres. However, as multi-classification is a somewhat more involved process, we will be limiting each movie to a single genre. The following commands accomplish our filtering needs:
```r
data_all <- subset(data_all, !duplicated(title)) # Single genre per movie
data_all <- subset(data_all, language == "English") # English only
data_all <- subset(data_all, country == "USA") # USA only
data_all <- subset(data_all, !grepl("(TV)|(V)|(VG)", title)) # No TV shows, straight-to-videos or video games
data_all <- subset(data_all, !grepl("Adult|Erotica", genre)) # No porn
data_all <- subset(data_all, note == "") # No regional edits
data_all <- subset(data_all, rcount >= 100) # Only movies with at least 100 reviews
```
##### Sampling
While we now have a workable data set, the records are still all in alphabetical order, which could skew the results of our modelling. As such, the final step is to randomize the records. It's also prudent to save a copy of these results in case we need to reproduce the same results later:
```r
data_rnd <- data_all[sample(nrow(data_all)),] # Randomize records
write.csv(data_rnd, "Clean-Random-Data.csv") # Save randomized records
```

### Data Modelling

##### Feature Selection
In typical modelling, models are applied to numerical data. Since we are working with text, however, we need to find to convert this text into meaningful numeric data, a process referred to as data mining. This process begins by selecting our *features*. In our project, features will mean *terms* from the *document*, that is words from a movie's plot, which are considered to be meaningful for classification. The process of feature selection will involve text mining all the *documents* in our *corpus*, that is all the movie plots from our database, which generally involves morphing and scoring the individual terms.

Morphing typically entails:
* Removing stop words, that is common words like "the" or "and" which appear too frequently to be meaningful;
* Stemming the words, that is cutting a word to its root such that "morphing" and "morphed" both become "morph";
* Removing other terms of limited interest, such a numbers and punctuation;
* and eliminating words which appear very infrequently, called *sparsity*.

Scoring is typically done by weighing the words using either TF, term frequency, or TF-IDF, term frequency-inverse document frequency. The former simply counts the frequency of terms within a document; the latter, however, increases a term's score the more frequently it appears within the document, while reducing the score the more frequently it appears in *other* documents, thereby giving you a more accurate representation of a term's importance in relation to the entire corpus.

In R, we can facilitate these tasks using any number of libraries. I have selected to use the *tm* and [*RTextTools*](http://www.rtexttools.com/documentation.html) libraries. Using our previously randomized data, we can create a weighing matrix like so:
```r
library("tm")
library("RTextTools")

data_matrix_tf <- create_matrix(data_rnd$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTf) # Create matrix
```
This creates a matrix, *data_matrix_tf*, which contains 18,898 rows, our movie list, and 3,262 columns, a list of weighted, stemmed words of interest which have become our features. Note that a sparsity of 0.998 was specified in order to reduce overhead. You can reduce this number to improve results at the expense of greater memory usage.

If we peek into our matrix, we can get a sense of what's been done:
```text
       Terms
Docs    divid divis divorc doc doctor document documentari doesnt dog doll dollar domest domin don donald done dont doom door dorothi
  1         0     0      0   0      0        0           0      0   0    0      1      0     0   0      0    0    0    0    0       0
  2         0     0      0   0      0        0           0      0   0    0      0      0     0   0      0    0    0    0    0       0
  3         0     0      0   0      0        0           0      0   0    0      1      0     0   0      0    0    0    0    0       0

       Terms
Docs    nativ natur navi navig nazi near nearbi necessari neck ned need neglect negoti neighbor neighborhood neil neither nelson nemesi
  1         0     0    0     0    0    0      0         0    0   0    0       0      0        0            0    0       0      0      0
  2         0     0    0     0    0    0      0         0    0   0    2       0      0        0            0    0       0      0      0
  3         0     0    0     0    0    0      0         0    0   0    0       0      0        0            0    0       0      0      0
```
Many terms have been stemmed, "divid" for example. We can see that "dollar" appears once in douments 1 and 3, and that "need" appears twice in document 2.

However, since TF is considered less representative than TD-IDF, we will create a new matrix using TD-IDF and compare the differences:
```r
data_matrix_tfidf <- create_matrix(data_rnd$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTfIdf) # Create matrix
```
Now we can see some subtle differences:
```text
       Terms
Docs        dollar     domest      domin        don     donald       done       dont       doom       door    dorothi      doubl
  1     0.10230835 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
  2     0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
  3     0.10912891 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000

       Terms
Docs          navi      navig       nazi       near     nearbi  necessari       neck        ned       need    neglect     negoti
  1     0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
  2     0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.24135401 0.00000000 0.00000000
  3     0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000 0.00000000
```
"dollar" still appears in documents 1 and 3, but has been given slightly less weight in document 1 than in document 3, likely because of the frequency in which it appears in any number of the remaining 18,895 documents not shown here. Similarly, "need" has been given a score in document 2 which better takes into account its frequency elsewhere.

It's important to note that both these functions create a *sparse* matrix. That is, if we have, as we've said, 18,898 documents with 3,262 features, we would normally require a matrix of 18,898 x 3,262 = 61,645,276 cells to store every possible value. However, since most of those values will be zero, we can use a sparse matrix to store only non-zero values and infer the rest, reducing the number of required cells in this case to 668,825 and lowering memory consumption considerably. This can be verified by viewing the matrix:
```r
data_matrix_tfidf
```
```text
<<DocumentTermMatrix (documents: 18898, terms: 3262)>>
Non-/sparse entries: 668825/60976451
Sparsity           : 99%
Maximal term length: 15
Weighting          : term frequency - inverse document frequency (normalized) (tf-idf)
```
We can write a quick little function that shows us what the highest scored features are for any given document:
```r
find_most_freq <- function(doc_num, how_many) {
  print(sort(as.matrix(data_matrix_tfidf)[doc_num,], decreasing=T)[1:how_many])
}
```
So if we wanted to see the top five features for the first document of our corpus, we can just run:
```r
find_most_freq(1,5)
```
```text
  compani  industri    enough  idealist       rat 
0.2501054 0.2131951 0.1725922 0.1366643 0.1337827 
```

### Training
With our matrix in hand, the next step is to create a data container, a means to tell R how to train and test our matrix data:
```r
data_container_tfidf <- create_container(data_matrix_tfidf, as.numeric(factor(data_rnd$genre)), trainSize=1:4000, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later
```
With this command, we are instructing R to take the matrix of our 18,898 movies and their features, and to model the data using our classes, in this case the genres of the individual movies, using the first 4000 records for training and the last 2,898 records for testing. Because classification is normally done on numeric values, we "factorize" the genres, that is we assign each of the 31 genres a number and use that number as the class.

If you peek into our container, you will now find two matrices, one for training and one for testing. As you might expect, both still have 3,262 columns, our features. Note that our 31 initial classes have been reduced to 24, either because they are never used or because we filtered them out earlier.

Again, these matrices are stored as sparse matrices, because of the high incidence of zero entries. However, we can visualize our sparse matrix to give a better sense of where our values exist, with each non-zero value represented by a grey dot:
```r
library("SparseM")
image(as.matrix.csr(data_container_tfidf@training_matrix), xlab="Terms", ylab="Documents")
title("TD-IDF Scoring of Terms by Document")
```
![Training Sparse Matrix Graph](/graphs/training-sparse-matrix.png)

More information on the performance advantages of sparse matrices can be found [here](http://www.johnmyleswhite.com/notebook/2011/10/31/using-sparse-matrices-in-r/).

### Support Vector Machine
With our container, we now ready to pick a model and begin training and classifying. I'll be using SVM, or Support Vector Machine, as a baseline algorithm for analysis. SVM was originally a linear classifier, but has been expanded to include non-linear functions. Two great tutorials on SVM can be found [here](https://lagunita.stanford.edu/c4x/HumanitiesandScience/StatLearning/asset/ch9.html) and [here](http://cbio.ensmp.fr/~jvert/svn/tutorials/practical/svmbasic/svmbasic_notes.pdf).

##### Hyperplanes
Understanding the mathematics behind SVM, which its [Wikipedia article](https://en.wikipedia.org/wiki/Support_vector_machine) summarizes, can be difficult without significant higher education in that field. But at its most basic level, it's important to understand that SVM works by attempting to map data to *hyperplanes*, that is, two-or-higher-dimensional grids. For the purpose of this explanation, we will be using simple two-dimensional, X and Y grids to try to outline the process. So, given any set of X and Y coordinates, mapping these on a grid would be easy. For example:

Doc|X|Y
---|---|---
1|0.3|0.6
2|0.4|0.7
3|0.6|0.2
4|0.1|0.5

![Simple XY plot](/graphs/simple-xy.png)

However, now take the following grid:

Doc|A|B|C|D
---|---|---|---|---
1|0.3|0.6|0.2|0.4
2|0.4|0.7|0.5|0.3
3|0.6|0.2|0.1|0.5
4|0.1|0.5|0.3|0.2

How would you graph this table? From a mathematical perspective, you could use a four-dimensional hyperplane. However, for us mere mortals, there is no practical way to visualize these points meaningfully. Accordingly, if we do want to plot our data, we must limit ourselves to two or three features. I'll be using two for the sake of clarity, which would look something like this:
```r
plot(as.matrix(data_container_tfidf@training_matrix)[,1:2], col = as.numeric(factor(data_rnd$genre)), xlab=data_container_tfidf@column_names[1], ylab=data_container_tfidf@column_names[2], title="One-versus-one plot of two first columns")
title("Plot of two first columns of training matrix")
```
![Plot of two first columns of training matrix](/graphs/plot-first-columns.png)

It's not particularly interesting, since most values are zero, and for our two first terms, *abbi* and *abandon*, there are no documents that are non-zero for both terms. So let's look for some more popular terms:
```r
head(sort(apply(as.matrix(data_container_tfidf@training_matrix), 2, function(c)sum(c!=0)), decreasing = T, index.return = T)$ix, 5)
```
```text
[1] 1231 1111 2022 1963 1693
```
Given my randomized data, these are indexes of the most populated terms. Your results will vary. So let's run that plot again using those indexes:
```r
plot(as.matrix(data_container_tfidf@training_matrix)[,c(1111,1231)], col = as.numeric(factor(data_rnd$genre)), xlab=data_container_tfidf@column_names[1111], ylab=data_container_tfidf@column_names[1231])
title("Plot of two most populated columns of training matrix")
```
![Plot of two most populated columns of training matrix](/graphs/plot-popular-columns.png)

This produces a much more visually interesting result. Now, while we can see from the rainbow of colours that all of our classes are displayed, this process isn't actually representative of how modelling works. Because SVM and similar functions operate primarily on binary classification, we must use either a *one-versus-one* or *one-versus-all* (also called *one-versus-many* or *one-versus-rest*) approach. In layman's terms, this just means that the modelling is run only on two classes at a time. So, in a one-versus-one approach of our example, you would first plot Action vs Adventure, and run your model on that, then Action vs Animation, Action vs Biography, and so on all the way through War vs Western, until you've gone through every possible combination (*not* permutation), and from those results, select the class selected by most classifiers.

##### Modelling
So because our wonderful computers can perform 3,262-dimensional hyperplane mathematics, let's compare both the original SVM linear function and the popular SVM Gaussian non-linear function, letting R decide the default parameters for each:
```r
train_SVM_linear <- train_model(data_container_tfidf, "SVM", kernel = "linear")
class_SVM_linear <- classify_model(data_container_tfidf, train_SVM_linear)

train_SVM_radial <- train_model(data_container_tfidf, "SVM", kernel = "radial") #Gaussian
class_SVM_radial <- classify_model(data_container_tfidf, train_SVM_radial)

head(cbind(class_SVM_linear, class_SVM_radial), 15)
```
```text
   SVM_LABEL  SVM_PROB SVM_LABEL  SVM_PROB
1          5 0.2703888         5 0.2386189
2         23 0.2036409        14 0.2975648
3          8 0.2447899         8 0.2886877
4          5 0.2812845         5 0.2658454
5         21 0.2463995        21 0.1769495
6         14 0.4506533        14 0.2748334
7          8 0.3280318         8 0.3149284
8          5 0.2132786         5 0.2263889
9         21 0.2752117         5 0.2310229
10         5 0.2304026         8 0.3085733
11        23 0.2359380        23 0.2126839
12        16 0.1768736         5 0.2409772
13         8 0.3900510         8 0.3704462
14         8 0.2682525         8 0.2469519
15         8 0.2760122         8 0.2949235
```
As we can see from the results, both kernels performed similarly, with mixed but generally low probabilities. We can also note that records were predominantly sorted into classes 5 and 8.

##### Class Biasing
According to this [research paper](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1527706&url=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D1527706), training sets with uneven class sizes results in biases towards classes with larger training sizes. So is that what is happening here? Let's take a look at the classes of our first 4,000 documents:
```r
table(head(data_rnd$genre, 4000))
```
```text

     Action   Adventure   Animation   Biography      Comedy       Crime Documentary       Drama      Family     Fantasy   Film-Noir 
        163         108          93          44         720         173         183         777         124          59          22 
  Game-Show     History      Horror       Music     Musical     Mystery        News     Romance      Sci-Fi       Short       Sport 
          0          33         268          49          68          91           3         286         115         171          38 
   Thriller         War     Western 
        266          37         109 
```
We can also visualize this data:
```r
# Histogram of genre distribution
data_block <- head(data_rnd$genre, 4000)
par(xaxt="n") # Turn off horizontal axis labels
plot(data_block, type="h", main = "Distribution of genres in first block") # Draw plot
par(xaxt="s") # Turn back on horizontal axis labels
axis(1, at=seq(par("xaxp")[1], par("xaxp")[2], by=(par("xaxp")[2]-par("xaxp")[1])/(length(unique(data_block))+1)), labels = FALSE) # Draw horizontal ticks
incr <- (par("xaxp")[2]-par("xaxp")[1])/(length(unique(data_block))+1) # Calculate distance between bars
text(x = seq(par("xaxp")[1]+incr, par("xaxp")[2], by=incr), y = -20, labels = sort(unique(data_rnd$genre)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8) # Write horizontal labels
```
![Distribution of genres in first block](/graphs/hist-genres.png)

As suspected, the 5th and 8th classes, Comedy and Drama, have much higher occurrences than any of the other classes. However, does that imply a biasing of our classification? It's possible given that, as we said earlier, in our one-versus-one approach, these two classes are most likely to be selected by the most classifiers. 

##### Results
So, overall, how well did we do?
```r
anals_SVM_linear <- create_analytics(data_container_tfidf, class_SVM_linear)
summary(anals_SVM_linear)
anals_SVM_radial <- create_analytics(data_container_tfidf, class_SVM_radial)
summary(anals_SVM_radial)
```
Kernel|Precision|Recall|FScore
---|---|---|---
Linear|0.2141667|0.1545833|0.1620833 
Gaussian|0.2079167|0.1529167|0.1541667 

Once again we can see that both kernels performed similarly, with a very slight advantage given to the linear classification. In both cases, our precision is approximately 21% and our recall is approximately 15%. But what's our overall accuracy?
```r
table(anals_SVM_linear@document_summary$SVM_LABEL == anals_SVM_linear@document_summary$MANUAL_CODE)
table(anals_SVM_radial@document_summary$SVM_LABEL == anals_SVM_radial@document_summary$MANUAL_CODE)
```
```text
FALSE  TRUE 
 2003   895 
 
FALSE  TRUE 
 1963   935 
```
Ultimately, our linear accuracy is 895 / 2898 = 0.3088337, or roughly 31%, compared to 32% for the Gaussian. At first glance, these numbers might seem low, but they must be contrasted against results from random chance, which would be approximately 1 in 24, or approximately 4.2%, given equal probabilities for all classes. We can visualize these results using a confusion matrix heatmap, using our linear kernel as an example, but because of some trickery with how R factorizes its labels, we must apply an unfactor function:
```r
unfactor <- function(obj) {
  unfactor <- as.numeric(levels(obj)[as.integer(obj)])
}

heatmap(table(unfactor(anals_SVM_linear@document_summary$SVM_LABEL), anals_SVM_linear@document_summary$MANUAL_CODE)[length(unique(unfactor(anals_SVM_linear@document_summary$SVM_LABEL))):1,], Rowv = NA, Colv = NA, col = heat.colors(256))
```
![Linear Kernel Confusion Matrix Heatmap](/graphs/heatmap-svm-linear.png)

The diagonal line of lighter cells across the square represents our true positives. Under normal circumstances, we could also plot and analyse the area under the curve (AUC); however, as explained earlier, a pairwise one-versus-one approach would be required to properly assess each class' performance, and multiclass ROC curves are still an evolving (and disputed) process. So we'll skip this for now.

##### Visualizing Classification
Instead, let's step back and once again try to understand what's happening under the hood here. We can use R's e1071 library to better step through SVM's linear performance.
```r
library(e1071)
data_for_svm <- data.frame(X1 = as.matrix(data_container_tfidf@training_matrix)[, 1111], X2 = as.matrix(data_container_tfidf@training_matrix)[, 1231], y = as.factor(data_rnd$genre[1:4000]))
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "linear", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Linear, all genres, all records", line = 0.6)
```
![SVM classification plot, linear, all genres, all records](/graphs/plot-linear-allg-allr.png)

If we look at the background colors of this plot, only two distinct colors appear, which means all the points are being biased into one of the two classes represented by these colors. The legend makes it a bit unclear which these are, as only four of the 24 values are written out, but it's a safe assumption that it's our usual two culprits, Comedy and Drama.

There are also some points in the extreme lower-left corner. These are points which are 0 for both terms in our plot. If we remember our sparse matrix graph from earlier, many zero entries exist, which could further bias our results by skewing the data towards them, particularly given their high frequency. So what happens if we remove these entries?
```r
data_for_svm <- data.frame(X1 = as.matrix(data_container_tfidf@training_matrix)[, 1111], X2 = as.matrix(data_container_tfidf@training_matrix)[, 1231], y = as.factor(data_rnd$genre[1:4000]))
data_for_svm <- subset(data_for_svm, X1 != 0 | X2 != 0) # Only keep values which are non-zero for one or both terms
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "linear", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Linear, all genres, non-zero records", line = 0.6)
```
![SVM classification plot, linear, all genres, non-zero records](/graphs/plot-linear-allg-nzr.png)

Without the zero entries, our classification separator looks better fit, but still biased to the same classes. Let's try one more time, isolating these two classes:
```r
genre_idxs <- which(head(data_rnd$genre, 4000) %in% c("Comedy", "Drama"))
data_for_svm <- data.frame(X1 = as.matrix(data_container_tfidf@training_matrix)[genre_idxs, 1111], X2 = as.matrix(data_container_tfidf@training_matrix)[genre_idxs, 1231], y = as.factor(data_rnd$genre[genre_idxs]))
data_for_svm <- subset(data_for_svm, X1 != 0 | X2 != 0)
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "linear", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Linear, comedies/dramas, non-zero records", line = 0.6)
```
![SVM classification plot, linear, comedies/dramas, non-zero records](/graphs/plot-linear-cd-nzr.png)

We can confirm now that our data is definitely being biased to these two classes. Although the background shading is easier to distinguish, results are basically identical. So what happens when we use a non-linear Gaussian function?
```r
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "radial", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Gaussian, comedies/dramas, non-zero records", line = 0.6)
```
![SVM classification plot, gaussian, comedies/dramas, non-zero records](/graphs/plot-radial-cd-nzr.png)

There is a negligable difference in performance, as we saw in our results earlier. Note that further refinements could be made by adjusting the cost factor *C*, the gamma and other kernel parameters that are typically optimized by the automated training functions.

However, now that we are only working with two classes, our model is a traditional binary classification problem and we can perform our classification to plot the area under the curve (AUC) using the *ROCR* R library, and visualize our results:
```r
library(ROCR)
data_for_svm$y = factor(data_for_svm$y) # Refactor class to remove unused classes
data_for_svm_pred <- predict(data_for_svm_fit, data_for_svm, probability = T, decision.values = T) # Run classification
table(data_for_svm_pred, data_for_svm$y) # Confusion Matrix

prob.comedy <- attr (data_for_svm_pred, "probabilities")[, "Comedy"] # Get Comedy probabilities
roc.pred <- prediction(prob.comedy, data_for_svm$y == "Comedy") # Get Comedy predictions and labels

data_for_svm_perf <- performance (roc.pred, "tpr", "fpr")
plot(data_for_svm_perf, main="ROC for SVM Linear Model")
abline(a = 0, b = 1, col="gray")
```
```text
data_for_svm_pred Comedy Drama
           Comedy    193   140
           Drama     128   164
```

![ROC for SVM Linear Binary Model](/graphs/roc-svm-linear-binary.png)

Our overall accuracy is now approximately 57%, however random chance is now 50% which makes our performance only marginally better, as can be seen on the ROC curve, with the gray line representing random chance.

At any rate, while this exercise has given us interesting insights from an academic perspective, it doesn't ultimately help us achieve our multiclass classification objectives. So how can we improve our results?

### Fitting The Data

##### Class distribution
Since we suspected that our results might be biased by our larger classes, let's build a new data set that attempts to better balance these. First, let's summarize the class distribution of our data:
```r
table(data_rnd$genre)
```
```text
     Action   Adventure   Animation   Biography      Comedy       Crime Documentary       Drama      Family     Fantasy   Film-Noir 
        802         514         392         221        3400         809         840        3804         594         250         125 
  Game-Show     History      Horror       Music     Musical     Mystery        News     Romance      Sci-Fi       Short       Sport 
          1         150        1239         247         286         393          14        1244         522         813         163 
   Thriller         War     Western 
       1329         209         537
```

##### Balancing
As *most* classes seem to have around 250 records or more, we can use that as our safe "bucket" size.
```r
# Create balanced data set with max. 250 records per class, and no bum classes
data_balanced <- NULL
for (g in 1:length(levels(data_rnd$genre))) {
  data_balanced <- rbind(data_balanced, head(subset(data_rnd, genre == levels(data_rnd$genre)[g]), 250))
}
data_balanced <- subset(data_balanced, !grepl("Game-Show|News", genre)) # Remove empty or near-empty classes
data_balanced$genre <- factor(data_balanced$genre) # Re-factorize
data_balanced <- data_balanced[sample(nrow(data_balanced)),] # Rando
```
Now let's see what our class distribution looks like:
```r
# Histogram of class distribution
par(xaxt="n") # Turn off horizontal axis labels
plot(data_balanced$genre, type="h", main = "Distribution of genres in balanced set") # Draw plot
par(xaxt="s") # Turn back on horizontal axis labels

incr <- par("usr")[2]/(length(levels(data_balanced$genre))+1) # Calculate distance between bars
axis(1, at=seq(0, par("usr")[2]-incr, by=incr), labels = FALSE) # Draw horizontal ticks
text(x = seq(0+incr, par("usr")[2]-incr, by=incr), y = -20, labels = sort(levels(data_balanced$genre)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8) # Write horizontal labels
```
![Distribution of genres in balanced set](/graphs/hist-genres-balanced.png)

##### Modelling
Now that we have a much more even distribution among classes, let's see what sort of results we can get:
```r
data_balanced_matrix_tfidf <- create_matrix(data_balanced$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTfIdf) # Create matrix
data_balanced_container_tfidf <- create_container(data_balanced_matrix_tfidf, as.numeric(factor(data_balanced$genre)), trainSize=1:4000, testSize = 4001:5365, virgin = FALSE) # Factorization of class is necessary or analysis will fail later

train_SVM_linear <- train_model(data_balanced_container_tfidf, "SVM", kernel = "linear")
class_SVM_linear <- classify_model(data_balanced_container_tfidf, train_SVM_linear)
anals_SVM_linear <- create_analytics(data_balanced_container_tfidf, class_SVM_linear)

summary(anals_SVM_linear)
table(unfactor(anals_SVM_linear@document_summary$SVM_LABEL), anals_SVM_linear@document_summary$MANUAL_CODE) # Confusion Matrix
table(anals_SVM_linear@document_summary$SVM_LABEL == anals_SVM_linear@document_summary$MANUAL_CODE) # Accuracy
```
```text
SVM_PRECISION    SVM_RECALL    SVM_FSCORE 
    0.2604348     0.2630435     0.2591304 
	
FALSE  TRUE 
 1002   363
```
With our balanced data set, we achieved a precision and recall of approximately 26%, an improvement over our unbalanced classes, but an overall accuracy of just under 27%, which is worse. So what could cause this? We can speculate that the predominance of classes 5 and 8 in our regular data set was actually helping our model by increasing the probability that any given document would fall into one of these two classes, making predictions "easier". With 23 even classes, any random guess has a lower probability of being true.

To better understand this, imagine that I'm playing a guessing game while blindfolded. I'm pulling jelly beans out of a big bowl and trying to guess their color, but I'm terrible at guessing so I always say red. Now if this bowl has 10 jelly beans each of a different color, then I only have a 1 in 10, or 10% chance of being correct. If, on the other hand, the bowl has 8 red jelly beans, 1 blue jelly bean and 1 green jelly bean, then suddenly I'll have an 8 in 10, or 80% chance of being correct. My color-picking skill hasn't actually improved any, I just have better chances of getting lucky. Thus the importance of balanced data.

![Heatmap for Balanced SVM Linear Model](/graphs/heatmap-svm-linear-balanced.png)

These are interesting results, but still not particularly useful. Given that SVM works best with two classes, and that our data might be *too* sparse for meaningful results, it may simply be that our data might be better suited to another model. So let's find out!

### LogitBoost
Another popular algorithm is LogitBoost, which claims to be less sensitive to outlier data than the algorithm on which it is based, AdaBoost, which takes an iterative voting-based methodology to classification. Outliers are generally unwelcome in statistics because they can skew the data towards them, and so best efforts should be made in excluding them.

Is our data prone to outliers and will Logitboost's insensitivity to them help our classification? Let's look at its performance in more detail and find out.

##### Manual Modelling
```r
library("caTools")
library("caret")

boost_model <- LogitBoost(head(as.matrix(data_matrix_tfidf), 4000), head(as.numeric(factor(data_rnd$genre)), 4000), nIter=10)
```
Here we instruct R to create our LogitBoost model on the first 4000 records of our weighed data. Typically, the higher the number of iterations, *nIter*, the more accurate the model will be. Next, we can run the prediction and see how our confidence levels:
```r
boost_scores <- predict(boost_model, tail(as.matrix(data_matrix_tfidf), 1000))
boost_scores_prob <- predict(boost_model, tail(as.matrix(data_matrix_tfidf), 1000), type = "raw")
t(cbind(boost_scores, round(boost_scores_prob, 4))[1:5,])
```
```text
               [,1]   [,2]   [,3]    [,4]   [,5]
boost_scores     NA     NA     NA 17.0000     NA
1            0.1192 0.0025 0.0180  0.0180 0.0180
2            0.0025 0.0180 0.1192  0.0025 0.0180
3            0.0180 0.0180 0.0180  0.0180 0.0180
4            0.0025 0.0025 0.0025  0.0025 0.0025
5            0.1192 0.1192 0.1192  0.1192 0.1192
6            0.0025 0.0180 0.0180  0.0180 0.0180
7            0.0025 0.0025 0.0025  0.0025 0.0025
8            0.1192 0.1192 0.0180  0.0180 0.1192
9            0.0180 0.0180 0.0180  0.0025 0.0180
10           0.0180 0.0180 0.5000  0.0180 0.0180
11           0.0025 0.0025 0.0025  0.0025 0.0025
13           0.0003 0.0025 0.0003  0.0025 0.0025
14           0.0180 0.1192 0.0180  0.1192 0.0180
15           0.0180 0.0025 0.0003  0.0025 0.0003
16           0.0025 0.0180 0.0180  0.0025 0.0025
17           0.0180 0.0180 0.5000  0.5000 0.0180
18           0.0000 0.0003 0.0003  0.0003 0.0003
19           0.1192 0.1192 0.1192  0.1192 0.1192
20           0.0180 0.0180 0.0180  0.0025 0.0025
21           0.0180 0.0180 0.0180  0.0180 0.0180
22           0.0025 0.0025 0.0025  0.0025 0.0025
23           0.1192 0.1192 0.1192  0.1192 0.1192
24           0.0025 0.0025 0.0025  0.0025 0.0025
25           0.0025 0.0180 0.0180  0.0180 0.0180
```
Note that I've transposed the results of this table using the *t* function for legibility, so that the documents are now the columns and the classes are now the rows. With that in mind, what this table allows us to see is the confidence, according to LogitBoost, of the first five documents falling into each of our 24 classes.

Differing levels of confidence can be observed. For example, LogitBoost has 50% confidence of document 4 belonging to class 17, the highest result for that document, and has therefore classified it as such, as seen by the 17 at the top of the column. On the other hand, there's 50% confidence that document 3 belongs to either 10 or 17, and no decision was made. The remaining documents have relatively poor confidence across the board, with no decisions either.

##### Results
We can gauge the overall accuracy of our model by visualizing a confusion matrix:
```r
table(boost_scores, tail(as.numeric(factor(data_rnd$genre)), 1000))
```
```text
boost_scores  1  2  3  4  5  6  7  8  9 10 11 13 14 15 16 17 18 19 20 21 22 23 24 25
          1   0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
          2   1  0  1  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
          3   0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
          5   1  0  0  0 17  0  0  9  1  1  0  0  0  1  2  1  0  7  1  5  0  1  0  0
          6   1  1  0  0  0  2  0  1  0  0  0  0  1  0  0  0  0  0  0  1  0  0  0  0
          7   0  0  0  3  4  0 18  5  1  0  0  2  2  2  0  0  1  0  0  4  0  1  0  0
          8   1  1  0  1  8  5  2 31  1  1  0  0  3  0  0  0  0  7  5  3  1  8  1  2
          9   0  0  1  0  1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  1  0  1  0  0
          14  1  0  0  0  0  1  0  1  0  0  0  0  6  1  0  1  0  1  0  1  0  4  0  2
          15  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1  0  0  0  0
          16  0  0  0  0  0  0  0  1  0  0  1  0  0  0  2  0  0  1  0  0  0  0  0  0
          17  0  1  1  0  1  0  0  6  0  0  0  0  2  0  0  2  0  0  0  0  0  9  0  0
          19  1  1  0  1  4  2  2 10  0  0  0  1  0  0  1  0  0  7  0  2  0  3  0  0
          20  0  0  0  0  0  0  0  1  1  0  0  0  2  0  0  0  0  0  5  0  0  1  0  0
          22  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
          23  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  2  0  0  0  0  0  1  0  0
          24  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
          25  0  0  1  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  7
```
Here we can see a number of true positives, particularly for classes 5, 7, 8, 14, 19 and 25, as well as a high number of incorrect classifications spread across every class. 99 of 299 classified documents were correct, which is a 33% accuracy rate, seemingly in line with SVM's performance. That said, we tested on 1,000 records, so why do we only have 299 results?

>Logitboost algorithm relies on a voting scheme to make classifications. Many (nIter of them)
week classifiers are applied to each sample and their findings are used as votes to make the final
classification. The class with the most votes "wins". However, with this scheme it is common for
two cases have a tie (the same number of votes), especially if number of iterations is even. In that
case NA is returned, instead of a label.

Well now, earlier we'll recall observing a document that had an equal probabability of falling into one of two classes, and no decision was made. In these cases, LogitBoost will simply deem the result inconclusive. For our data, 701 documents were discarded as such, which is certainly not ideal!

As was mentioned earlier, this algorithm takes an iterative voting approach to classification, which means that is uses [Backpropagation](https://en.wikipedia.org/wiki/Backpropagation) to improve its results. In layman's terms, it measures and compares its errors levels through each training iteration and adjusts the parameters to adjust and compensate. In our example, each extra iteration should reduce the likelihood of a tie, so let's increase the iterations to 20 and try again:
```text
boost_scores  1  2  3  4  5  6  7  8  9 10 11 13 14 15 16 17 18 19 20 21 22 23 24 25
          1   3  0  0  0  0  1  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
          2   1  0  1  0  1  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0
          3   0  0  0  0  2  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0
          5   2  0  1  0 16  0  0  7  2  0  0  0  2  0  0  1  0  5  1  4  0  2  0  0
          6   3  1  0  0  0  2  0  2  0  0  0  0  2  0  0  1  0  0  0  0  0  4  0  1
          7   1  0  0  3  2  0 21  3  0  0  0  2  3  2  0  0  1  0  0  4  0  0  0  0
          8   3  1  0  1  7  2  2 16  0  0  0  0  1  0  0  0  0  3  0  1  0  6  0  1
          9   0  0  1  0  1  0  0  1  2  1  0  0  0  0  0  0  0  0  0  2  0  1  0  0
          10  1  0  0  0  1  0  0  0  0  0  0  1  1  0  0  0  0  0  0  0  1  0  0  0
          11  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
          13  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
          14  1  0  0  0  2  1  0  3  0  0  0  0  1  1  0  1  0  1  0  0  0  5  1  0
          15  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  1  0  0  0  0
          16  0  1  0  0  0  0  0  2  0  0  0  0  0  1  0  0  0  1  0  0  0  0  0  0
          17  0  0  0  0  1  0  0  3  0  0  0  0  3  0  0  1  0  0  0  0  0  4  0  0
          19  1  3  0  0 13  3  1 22  1  0  1  1  2  0  2  0  0 18  1  3  1  2  0  1
          20  0  0  0  0  0  0  0  2  0  0  0  0  3  0  0  0  0  0  4  0  0  1  0  0
          21  1  0  1  0  2  0  2  0  1  0  0  0  0  0  0  0  0  1  0  2  0  1  0  0
          22  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0
          23  0  0  0  1  0  2  1  6  0  0  0  0  3  0  0  2  0  0  0  0  0  0  0  0
          24  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0
          25  0  0  1  0  1  0  0  3  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0 10
```
Here we see better representation among the classes, with slightly more documents being classified, but our overall result of 97 of 342 documents, or 28%, is actually a deterioration in our accuracy. Is it likely that more iterations would lead to more classifications but lower accuracy? More testing will be required to confirm that hypothesis.

##### Assisted Modelling
Perhaps RTextTools does a better job of optimizing algorithms. Let's examine its results more closely:
```r
data_model <- train_model(data_container_tfidf, algorithm = "BOOSTING")
data_results <- classify_model(data_container_tfidf, data_model)
data_analytics <- create_analytics(data_container_tfidf, data_results)
```
In examining *data_model*, we can actually see that RTextTools has opted for 100 iterations. So let's map the confusion matrix again:
```r
table(unfactor(data_results$LOGITBOOST_LABEL), tail(as.numeric(factor(data_rnd$genre)), 2898))
```
```text
       1   2   3   4   5   6   7   8   9  10  11  13  14  15  16  17  18  19  20  21  22  23  24  25
  1   13   3   3   0  24  10   6  29   8   5   0   1   6   3   5   5   0   8   4   3   1  22   5   5
  2   21  20   7   3  26   9  11  33   6   2   0   2   8   2   6   3   0   9   8   9   0  17   3   7
  3    2   2   9   0  20   4   3  10  12   2   0   1   3   0   3   1   0   2   1  16   0   3   0   0
  4    0   1   0   1   1   0   1   7   1   0   0   0   0   0   0   0   0   1   0   0   1   0   0   0
  5   32  10  15   9 228  24  24 197  35  11   5   5  41  14  15   8   0  78  17  42   9  44   1  18
  6   13   7   3   2  30  23   2  54   1   0   2   2  11   3   0   7   0   9   2   3   0  27   0  13
  7    2   1   0   8  10   3  58  25   1   1   0   6  11   7   0   1   4   3   2   5   0   6   4   0
  8   18  16   2  11  78  18  15 140   8   5   6   5  18   4   5   6   0  37  11  10   4  30   7  17
  9    2   1   4   0   7   2   1   5   2   0   0   0   6   1   0   0   0   1   1   6   0   1   0   0
  10   3   3   0   0   4   0   0   5   0   1   0   0   3   0   0   0   0   3   0   1   0   4   0   1
  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0
  13   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0
  14   2   3   2   0  16   6   2  12   0   2   0   1  32   0   1   6   0   2   2   1   0  22   0   0
  15   1   1   0   1   6   0   1   4   1   0   0   0   1   5   2   1   0   3   1   5   0   2   0   0
  16   2   1   1   0   8   0   1   6   4   0   0   0   2   1   3   1   0   3   0   3   0   1   0   2
  17   1   0   3   0   1   3   1   8   1   1   0   0   9   0   1   3   0   3   2   0   0   8   0   1
  19   3   6   0   0  10   4   1  23   0   2   0   3   0   1   4   2   0  18   2   3   0   2   0   0
  20   3   2   1   0   4   0   2   3   2   1   0   0   4   0   0   0   0   0  12   2   0   3   0   0
  21   1   0   6   0  12   0   6   4   5   0   1   1   2   2   2   0   0   3   3  10   1   1   1   1
  22   0   0   1   1   2   0   1   1   1   0   0   0   2   0   0   0   0   0   0   0   4   0   0   0
  23  10   3   2   1  13   9   4  22   3   2   1   1  19   0   0   5   0   0   7   6   0  18   4   4
  24   3   1   0   0   1   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   1   2   0
  25   2   1   1   0   3   0   0   6   2   0   0   0   1   0   0   0   0   1   1   1   0   1   1  18
```
With 100 iterations, all 2,898 of our testing documents were successfully classified, with 620 true positives for an overall accuracy of approximately 21%. Not bad, but worse than SVM.

##### Biasing
Does LogitBoost perform any better with balanced classes? Let's model and see:
```r
data_balanced_model <- train_model(data_balanced_container_tfidf, algorithm = "BOOSTING")
data_balanced_results <- classify_model(data_balanced_container_tfidf, data_balanced_model)
data_balanced_analytics <- create_analytics(data_balanced_container_tfidf, data_balanced_results)

summary(data_balanced_analytics)
table(data_balanced_analytics@document_summary$LOGITBOOST_LABEL == data_balanced_analytics@document_summary$MANUAL_CODE) # Accuracy
```
```text
LOGITBOOST_PRECISION    LOGITBOOST_RECALL    LOGITBOOST_FSCORE 
           0.2052174            0.1721739            0.1773913 
		   
FALSE  TRUE 
 1132   233 
```
Well, no. Our results are worse still, with an accuracy of approximately 17%. As with before, having equal classes actually reduces the chance of a correct guess.

### Expanding our Analysis
So far, we've tried two algorithms with various settings, on differently sized and balanced data sets. Our results can be summarized as follows:

Model|Settings|Training|Testing|Class Sizes|Classes|Coverage|Precision|Recall|F-Score|Accuracy
---|---|---|---|---|---|---|---|---|---|---
SVM|Linear|4,000|2,898|0 - 777|25|100%|0.214|0.155|0.162|0.309
SVM|Gaussian|4,000|2,898|0 - 777|25|100%|0.214|0.155|0.162|0.323
SVM|Linear|4,000|1,365|125 - 250|23|100%|0.260|0.263|0.259|0.266
SVM|Gaussian|4,000|1,365|125 - 250|23|100%|0.199|0.207|0.144|0.185
LogitBoost|10 i.|4,000|1,000|0 - 777|25|30%|-|-|-|0.331
LogitBoost|20 i.|4,000|1,000|0 - 777|25|34%|-|-|-|0.284
LogitBoost|100 i.|4,000|1,000|0 - 777|25|100%|-|-|-|0.214
LogitBoost|100 i.|4,000|1,365|125 - 250|23|100%|0.205|0.172|0.177|0.171

Overall, Gaussian SVM with large training sets is the current frontrunner. However, we can include additional models and algorithms, which RTextTools makes very easy to execute. The following code divides our 18,898 records into four blocks of 4,000 documents, with the remaining documents used for testing. Each block of documents is trained and classified sequentially, and the default parameters for each algorithm are used.
```r
for (idx in c(1:4000, 4001:8000, 8001:12000, 12001:16000)) {

  data_container_tfidf <- create_container(data_matrix_tfidf, as.numeric(factor(data_rnd$genre)), trainSize=idx, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later
  
  data_models <- train_models(data_container_tfidf, algorithms = c("SVM", "MAXENT", "SLDA", "BOOSTING"))
  data_results <- classify_models(data_container_tfidf, data_models)
  data_analytics <- create_analytics(data_container_tfidf, data_results)
  
  summary(data_analytics)
}
```
If we collect the accuracy results from each block of documents for each algorithm, we end up with the following results:

Algorithm\Docs|1 - 4000|4001 - 8000|8001 - 12000|12001 - 16000
---|---|---|---|---
SVM|0.312|0.315|0.321|0.312
SLDA|0.277|0.288|0.295|0.286
MAXENT|0.242|0.269|0.267|0.259
LOGITBOOST|0.209|0.222|0.225|0.227

These results are pretty consistent with our own findings, with SVM still in the lead. We can also visualize this data:
```r
par(xaxt="n") # Turn off horizontal axis labels
plot(type = "o", x = c(1:4), y = subset(auto_results, V2 == unique(auto_results$V2)[1])$V3, ylim=c(0.2, 0.45), col=c("red", "blue", "green", "orange")[1], ylab = "Accuracy", xlab="Sequential Documents", main = "Sequential Classification Accuracy", pch = 19)
for (i in 2:length(unique(auto_results$V2))) {
  points(type = "o", x = c(1:4), y = subset(auto_results, V2 == unique(auto_results$V2)[i])$V3, col=c("red", "blue", "green", "orange")[i], pch = 19)
}
par(xaxt="s") 
axis(1, 1:4, unique(auto_results$V1)) # Draw horizontal ticks and labels
legend("topright", legend = unique(auto_results$V2), col=c("red", "blue", "green", "orange"), text.col=c("red", "blue", "green", "orange"), cex=0.8, pch=19)
```

![Sequential Classification Accuracy](/graphs/plot-accuracy-sequential.png)

We can observe that different document blocks produce marginally different results. However, it's generally considered to be best practice to train on cumulative documents, so let's see what happens if we do that. Please note that this is a very computationally expensive task, so make sure your computer is up to par before attempting it.

Algorithm\Docs|1 - 4000|1 - 8000|1 - 12000|1 - 16000
---|---|---|---|---
SVM|0.326|0.343|0.361|0.367
SLDA|0.295|0.321|0.337|0.349
MAXENT|0.268|0.259|0.272|0.255
LOGITBOOST|0.214|0.217|0.233|0.233

We can observe a nominal but proportional increase in accuracy with respect to the number of training documents for most algorithms:

![Cumulative Classification Accuracy](/graphs/plot-accuracy-cumulative.png)

And our heatmaps trained with 16,000 documents:

![Heatmap, SVM, 16000 documents](/graphs/heatmap-final-svm.png)
![Heatmap, SLDA, 16000 documents](/graphs/heatmap-final-slda.png)
![Heatmap, MAXENT, 16000 documents](/graphs/heatmap-final-maxent.png)
![Heatmap, LOGITBOOST, 16000 documents](/graphs/heatmap-final-logit.png)

##### Results
If we continue testing the remainder of our algorithms, we can achieve results similar to this:

Algorithm|Training Docs|Precision|Recall|F-Score|Accuracy
---|---|---|---|---|---
Bootstrap Aggregation|4000|0.11583333|0.07250000|0.06208333|0.2456866
Bootstrap Aggregation|16000|0.16208333|0.09208333|0.08750000|0.2684610
Generalized Linear Models|4000|0.20166667|0.09791667|0.09875000|0.2791580
LogitBoost|4000|0.1595833|0.1325000|0.1345833|0.2139406
LogitBoost|8000|0.1700000|0.1470833|0.1450000|0.2170462
LogitBoost|12000|0.2191667|0.1579167|0.1612500|0.2325741
LogitBoost|16000|0.2012500|0.1529167|0.1508333|0.2329192
Maximum Entropy|4000|0.1929167|0.1633333|0.1725000|0.2681159
Maximum Entropy|8000|0.1825000|0.1687500|0.1716667|0.2587991
Maximum Entropy|12000|0.1991667|0.1845833|0.1887500|0.2715665
Maximum Entropy|16000|0.1954167|0.1729167|0.1795833|0.2546583
Neural Network|4000|0.00875000|0.04166667|0.01458333|0.2056590
Neural Network|16000|0.02458333|0.06250000|0.03208333|0.2142857
Random Forest|4000|0.15500000|0.10416667|0.09541667|0.3015873
Random Forest|16000|0.2825000|0.1283333|0.1237500|0.3178053
Supervised Latent Dirichlet Allocation|4000|0.2262500|0.1737500|0.1887500|0.2950310
Supervised Latent Dirichlet Allocation|8000|0.2483333|0.2212500|0.2287500|0.3209109
Supervised Latent Dirichlet Allocation|12000|0.2566667|0.2462500|0.2458333|0.3374741
Supervised Latent Dirichlet Allocation|16000|0.2658333|0.2566667|0.2558333|0.3488612
Support Vector Machine|4000|0.2179167|0.1495833|0.1504167|0.3264320
Support Vector Machine|8000|0.2254167|0.1770833|0.1775000|0.3429951
Support Vector Machine|12000|0.2583333|0.2008333|0.2033333|0.3605935
Support Vector Machine|16000|0.2695833|0.2179167|0.2220833|0.3671497
Tree|4000|0.03958333|0.05750000|0.03708333|0.2091097
Tree|16000|0.03125000|0.05291667|0.03000000|0.2177363

For which the accuracy can be better visualized as follows:

![Overall Accuracy per Algorithm](/graphs/algo-performance.png)

##### k-Fold Cross-Validation
Cross validation is another technique for assessing how our models will perform on independent data sets. Unlike our previous examples, in which we trained and tested on separate data, cross-validation divides the data set into *k* folds, testing on 1 fold and training on the remaining k - 1 folds, incrementally changing the test fold until all folds have been covered.

Testing a number of our algorithms using 5-fold cross-validation, we achieve the following results:

![5-fold Cross-Validation Results](/graphs/cf-results.png)

While the last four algorithms all perform roughly equivalently to their random data testing performance, the first four showed noticeable differences. LogitBoost, GLMNet and MaxEnt may all be good examples of *over-fitting*. That is, because the models are too strongly fit around the training data, they test extremely well on this same data, but poorly on any new data that was previously unseen. Neural Network, on the other hand, may be a case of mild underfitting, essentially the opposite problem.

### Conclusion

Multiclass classification is a challenging process, all the more so when combined with text mining. Our experiments have shown that Support Vector Machine demonstrated overall the best performance, particularly with larger training sets.

With enough training records, it may be possible to increase the overall accuracy to a level which would be useful for practical predictions. However, the law of diminishing returns is likely to apply once a certain threshold is reached. In some cases, we also observed certain algorithms, such as Maximum Entropy, beginning to perform worse with extremely large training sets, suggesting that plateauing is a likely outcome.

Ultimately, with our data in its current state, automatic classification of genre based on plot may not be achievable. That said, re-examining and restructuring the data would be a better approach towards achieving better accuracy. With adjusted sparsity, as well as the possible inclusion of additional features unrelated to plot, it may be possible to build a high-performing model. Further experimentation is needed.

### Further Research

The following tasks may be worth pursuing to further improve results:

* Adjust algorithm parameters to prevent overfitting.
* Manipulate sparsity levels and determine affect on weighted matrices.
* Attempt different block sizes and determine affect on results.
* Score and include movie title to modelling to determine impact on weighing.
* Include additional non-semantic features, including movie length, year, and so forth to determine their relevance.
* Test Generalized Linear Models *(GLMNet)* with 16,000 records, which requires more than 16GB of RAM and failed during my tests.
* Investigate e1071's [tune](http://www.inside-r.org/packages/cran/e1071/docs/tune) function.

### Reference R Library Manuals

* [RTextTools manual](https://cran.r-project.org/web/packages/RTextTools/RTextTools.pdf)
* [e1071 manual](https://cran.r-project.org/web/packages/e1071/e1071.pdf)
* [caTools manual](https://cran.r-project.org/web/packages/caTools/caTools.pdf)
* [caret manual](https://cran.r-project.org/web/packages/caret/caret.pdf)
* [SparseM manual](https://cran.r-project.org/web/packages/SparseM/SparseM.pdf)
* [ROCR manual](https://cran.r-project.org/web/packages/ROCR/ROCR.pdf)
* [AUC manual](https://cran.r-project.org/web/packages/AUC/AUC.pdf)