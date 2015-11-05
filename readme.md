# What's in a name?
##### A machine learning and text mining experiment
by Jonathan Mikhail

### Background

Ask fans to describe the plot of “The Dark Knight”, “Die Hard” or “Scarface”, and they can usually do so quickly and comprehensively. But are movie plots simply a random description of a story, divorced from its genre, or is there a pattern which correlates these features? I would like to perform analysis to determine whether such a correlation exists, whether a classification model can be built from it, and whether this model can be used to predict a movie’s genre based only on its plot.

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

### Data Source

To perform my analysis, I’ve used the September 11, 2015 release of the Internet Movie Database (IMDB), the latest version of which is available for download [here](ftp://ftp.fu-berlin.de/pub/misc/movies/database/). The data is in compressed plain text, column aligned or tab delineated and prefaced with one or more introductory sections, with each file containing one major movie attribute.

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
> data_ratings <- read.delim("./GitHub/whatsinaname/data/cleaned/ratings.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("rcount","rating","title"))
> data_languages <- read.delim("./GitHub/whatsinaname/data/cleaned/language.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","language","variant"))
> data_countries <- read.delim("./GitHub/whatsinaname/data/cleaned/countries.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","country"))
> data_running_times <- read.delim("./GitHub/whatsinaname/data/cleaned/running-times.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","time","note"))
> data_genres <- read.delim("./GitHub/whatsinaname/data/cleaned/genres.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","genre"))
> data_plot <- read.delim("./GitHub/whatsinaname/data/cleaned/plot.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","plot"))
```

### Data Structuring

##### Merging
Since R creates a separate data frame for each imported file, we can now merge these into a unified data frame by matching on title:
```r
> data_all <- data.frame(data_ratings["title"], data_ratings["rating"], data_ratings["rcount"])
> data_all <- merge(data_all, data_running_times, by.x = "title", by.y = "title")
> data_all <- merge(data_all, data_languages, by.x = "title", by.y = "title")
> data_all <- merge(data_all, data_countries, by.x = "title", by.y = "title")
> data_all <- merge(data_all, data_genres, by.x = "title", by.y = "title")
> data_all <- merge(data_all, data_plot, by.x = "title", by.y = "title")
```
##### Filtering
Our merges have produced a sizeable data frame, however not all the data contained is pertinent. For simplicity's sake, we would like to restrict our analysis to English language, American-produced theatrical films, with at least 100 ratings.

Another important note is that movies are categorized into several different genres. However, as multi-classification is a somewhat more involved process, we will be limiting each movie to a single genre. The following commands accomplish our filtering needs:
```r
> data_all <- subset(data_all, !duplicated(title)) # Single genre per movie
> data_all <- subset(data_all, language == "English") # English only
> data_all <- subset(data_all, country == "USA") # USA only
> data_all <- subset(data_all, !grepl("(TV)|(V)|(VG)", title)) # No TV shows, straight-to-videos or video games
> data_all <- subset(data_all, !grepl("Adult|Erotica", genre)) # No porn
> data_all <- subset(data_all, note == "") # No regional edits
> data_all <- subset(data_all, rcount >= 100) # Only movies with at least 100 reviews
```
##### Sampling
While we now have a workable data set, the records are still all in alphabetical order, which could skew the results of our modelling. As such, the final step is to randomize the records. It's also prudent to save a copy of these results in case we need to reproduce the same results later:
```r
> data_rnd <- data_all[sample(nrow(data_all)),] # Randomize records
> write.csv(data_rnd, "Clean-Random-Data.csv") # Save randomized records
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

In R, we can facilitate these tasks using any number of libraries. I have selected to use the **tm** and [**RTextTools**](http://www.rtexttools.com/documentation.html) libraries. Using our previously randomized data, we can create a weighing matrix like so:
```r
> library("tm")
> library("RTextTools")

data_matrix_tf <- create_matrix(data_rnd$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTf) # Create matrix
```
This creates a matrix, *data_matrix_tf*, which contains 18,898 rows, our movie list, and 3,262 columns, a list of weighted, stemmed words of interest which have become our features. Note that a sparsity of 0.998 in order to reduce overhead. You can reduce this number to improve results at the expense of greater memory usage.

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
> data_matrix_tfidf <- create_matrix(data_rnd$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTfIdf) # Create matrix
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
"dollar" still appears in documents 1 and 3, but has been giving slightly less weight in document 1 than in document 3, likely because of the frequency in which it appears in any number of the remaining 18,895 documents not shown here. Similarly, "need" has been giving a score in document 2 which better takes into account its frequency elsewhere.

It's important to note that both these functions create a *sparse* matrix. That is, if we have, as we've said, 18,898 documents with 3,262 features, we would normally require a matrix of 18,898 x 3,262 = 61,645,276 cells to store every possible value. However, since most of those values will be zero, we can use a sparse matrix to store only non-zero values and infer the rest, reducing the number of required cells in this case to 668,825 and lowering memory consumption considerably. This can be verified by viewing the matrix:
```r
> data_matrix_tfidf
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
> find_most_freq(1,5)
```
```text
  compani  industri    enough  idealist       rat 
0.2501054 0.2131951 0.1725922 0.1366643 0.1337827 
```

### Training
With our matrix in hand, the next step is to create a data container, a means to tell R how to train and test our matrix data:
```r
> data_container_tfidf <- create_container(data_matrix_tfidf, as.numeric(factor(data_rnd$genre)), trainSize=1:4000, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later
```
With this command, we are instructing R to take the matrix of our 18,898 movies and their features, and to model the data using our classes, in this case the genres of the individual movies, using the first 4000 records for training and the last 2,898 records for testing. Because classification is normally done on numeric values, we "factorize" the genres, that is we assign each of the 31 genres a number and use that number as the class.

If you peek into our container, you will now find two matrices, one for training and one for testing. As you might expect, both still have 3,262 columns, our features. Note that our 31 initial classes have been reduced to 24, either because they are never used or because we filtered them out earlier.

Again, these matrices are stored as sparse matrices, because of the high incidence of zero entries. However, we can visualize our sparse matrix to give a better sense of where our values exist, with each non-zero value represented by a grey dot:
```r
> library("SparseM")
> image(as.matrix.csr(data_container_tfidf@training_matrix), xlab="Terms", ylab="Documents")
> title("TD-IDF Scoring of Terms by Document")
```
![Training Sparse Matrix Graph](/graphs/training-sparse-matrix.png)

More information on the performance advantages of sparse matrices can be found [here](http://www.johnmyleswhite.com/notebook/2011/10/31/using-sparse-matrices-in-r/).

### Support Vector Machine
With our container, we now ready to pick a model and begin training and classifying. I'll be using SVM, or Support Vector Machine, as a baseline model for analysis. SVM was originally a linear classifier, but has been expanded to include non-linear functions. Two great tutorials on SVM can be found [here](https://lagunita.stanford.edu/c4x/HumanitiesandScience/StatLearning/asset/ch9.html) and [here](http://cbio.ensmp.fr/~jvert/svn/tutorials/practical/svmbasic/svmbasic_notes.pdf).

Understanding the mathematics behind SVM, which its [Wikipedia article](https://en.wikipedia.org/wiki/Support_vector_machine) summarizes, can be difficult without significant higher education in that field. But at its most basic level, it's important to understand that SVM works by attempting to map data to *hyperplanes*, that is, two-or-higher-dimensional grids. For the purpose of this explanation, we will be using simple two-dimensional, X and Y grids to try to outline the process. So, given any set of X and Y coordinates, mapping these on a grid would be easy. For example:

Doc|X|Y
---|---|---
1|0.3|0.6
2|0.4|0.7
3|0.6|0.2
4|0.1|0.5

![Simple XY plot](/graph/simple-xy.png)

However, because we are working with matrices, SVM and similar functions must use either a *one-versus-one* or *one-versus-all* (also called *one-versus-many* or *one-versus-rest*) approach. To better understand that, take the following grid:

Doc|A|B|C|D
---|---|---|---|---
1|0.3|0.6|0.2|0.4
2|0.4|0.7|0.5|0.3
3|0.6|0.2|0.1|0.5
4|0.1|0.5|0.3|0.2

How would you graph this table? From a mathematical perspective, you could use a four-dimensional hyperplane. However, for us mere mortals, there is no practical way to visualize these points meaningfully. So in a one-versus-one approach, you would first plot A vs B, and run your model on that, then A vs C, A vs D, B vs C and so on until you've gone through every combination, and from those results, select the class selected by most classifiers. On a matrix like ours, with 3,262 columns, this would be an incredibly time-consuming task, requiring 5,318,691 plots. But let's take a look at what one of those plots might look like:
```r
plot(as.matrix(data_container_tfidf@training_matrix)[,1:2], col = as.numeric(factor(data_rnd$genre)), xlab=data_container_tfidf@column_names[1], ylab=data_container_tfidf@column_names[2], title="One-versus-one plot of two first columns")
title("Plot of two first columns of training matrix")
```
![Plot of two first columns of training matrix](/graphs/plot-first-columns.png)

It's not particularly interesting, since most values are zero, and for our two first terms, *abbi* and *abandon*, there are no documents that are non-zero for both terms. So let's look for some more popular terms:
```r
> head(sort(apply(as.matrix(data_container_tfidf@training_matrix), 2, function(c)sum(c!=0)), decreasing = T, index.return = T)$ix, 5)
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

This produces a much more interesting result, but given the multiple classes and divergent data, can suggest some of the challenges in applying SVM to our data. Nevertheless, computers are experts at repetitive, time-consuming tasks, and so let's compare both the original SVM linear function and the popular SVM Gaussian non-linear function, letting R decide the default parameters for each:
```r
> train_SVM_linear <- train_model(data_container_tfidf, "SVM", kernel = "linear")
> class_SVM_linear <- classify_model(data_container_tfidf, train_SVM_linear)

> train_SVM_radial <- train_model(data_container, "SVM", kernel = "radial") #Gaussian
> class_SVM_radial <- classify_model(data_container_tfidf, train_SVM_radial)

> head(cbind(class_SVM_linear, class_SVM_radial), 15)
```
```text
   SVM_LABEL  SVM_PROB SVM_LABEL  SVM_PROB
1          5 0.2680421         5 0.2576749
2         23 0.2048374         5 0.2476217
3          8 0.2434515         5 0.2394585
4          5 0.2726620         5 0.2521533
5         21 0.2418776         5 0.2596507
6         14 0.4467603         5 0.2341590
7          8 0.3230210         5 0.2300045
8          5 0.2197894         5 0.2518738
9         21 0.2466669         5 0.2493218
10         5 0.2321563         5 0.2484624
11        23 0.2348757         5 0.2307431
12         8 0.1861349         5 0.2629878
13         8 0.3796116         5 0.2469428
14         8 0.2626149         5 0.2461823
15         8 0.2752718         5 0.2558068
```
As we can see from the results, the linear function, shown in the first two columns, classified a variety of classes, albeit with relatively low confidence. The Gaussian function, however, classified every document in our sample to the same class, with little improvement in confidence.
```r
> unique(class_SVM_linear$SVM_LABEL)
> unique(class_SVM_radial$SVM_LABEL)
```
```text
[1] 5  23 8  21 14 7  1  24 20 25 15 6  9  3  4  2  22 19 17 16 10 11
Levels: 1 10 11 14 15 16 17 19 2 20 21 22 23 24 25 3 4 5 6 7 8 9

[1] 5 8
Levels: 5 8
```
In fact, while the linear function used nearly every class, the Gaussian function classified everything to one of two classes, 5 and 8. This is obviously not ideal, but it is expected; according to this [research paper](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=1527706&url=http%3A%2F%2Fieeexplore.ieee.org%2Fstamp%2Fstamp.jsp%3Ftp%3D%26arnumber%3D1527706), training sets with uneven class sizes results in biases towards classes with larger training sizes. Let's take a look at the class of our first 4,000 documents:
```r
> table(head(data_rnd$genre, 4000))
```
```text

     Action   Adventure   Animation   Biography      Comedy       Crime Documentary       Drama      Family     Fantasy   Film-Noir 
        163         108          93          44         720         173         183         777         124          59          22 
  Game-Show     History      Horror       Music     Musical     Mystery        News     Romance      Sci-Fi       Short       Sport 
          0          33         268          49          68          91           3         286         115         171          38 
   Thriller         War     Western 
        266          37         109 
```
As expected, the 5th and 8th classes, Comedy and Drama, have much higher occurrences than any of the other classes, which explains the biasing of our Gaussian function. So let's step back and once again try to understand what's happening under the hood here. We can use R's e1071 library to better step through SVM's linear performance.
```r
library(e1071)
data_for_svm <- data.frame(X1 = as.matrix(data_container_tfidf@training_matrix)[, 1111], X2 = as.matrix(data_container_tfidf@training_matrix)[, 1231], y = as.factor(data_rnd$genre[1:4000]))
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "linear", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Linear, all genres, all records", line = 0.6)
```
![SVM classification plot, linear, all genres, all records](/graphs/plot-linear-allg-allr.png)

We can in the color shading that the results have been biased to the same two classes. As well, if we remember our sparse matrix graph from earlier, many entries exist that are zero for both terms, which could further bias our results. So what happens if we remove these entries?
```r
data_for_svm <- data.frame(X1 = as.matrix(data_container_tfidf@training_matrix)[, 1111], X2 = as.matrix(data_container_tfidf@training_matrix)[, 1231], y = as.factor(data_rnd$genre[1:4000]))
data_for_svm <- subset(data_for_svm, X1 != 0 | X2 != 0)
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "linear", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Linear, all genres, non-zero records", line = 0.6)
```
![SVM classification plot, linear, all genres, non-zero records](/graphs/plot-linear-allg-nzr.png)

Our classification looks a little better here, but is still biased to the same classes. Let's try one more time, isolating these two classes:
```r
genre_idxs <- which(head(data_rnd$genre, 4000) %in% c("Comedy", "Drama"))
data_for_svm <- data.frame(X1 = as.matrix(data_container_tfidf@training_matrix)[genre_idxs, 1111], X2 = as.matrix(data_container_tfidf@training_matrix)[genre_idxs, 1231], y = as.factor(data_rnd$genre[genre_idxs]))
data_for_svm <- subset(data_for_svm, X1 != 0 | X2 != 0)
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "linear", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Linear, comedies/dramas, non-zero records", line = 0.6)
```
![SVM classification plot, linear, comedies/dramas, non-zero records](/graphs/plot-linear-cd-nzr.png)

Although easier to read, results are basically identical. So what happens when we use a non-linear Gaussian function?
```r
data_for_svm_fit = svm(factor(y) ~ ., data = data_for_svm, scale = FALSE, kernel = "radial", cost = 10)
plot(data_for_svm_fit, data_for_svm)
mtext(side = 3, text = "Gaussian, comedies/dramas, non-zero records", line = 0.6)
```
![SVM classification plot, gaussian, comedies/dramas, non-zero records](/graphs/plot-radial-cd-nzr.png)

Nearly identical results. Note that further refinements could be made by adjusting the cost factor *C*, the gamma and other kernel parameters that are typically optimized by the automated training functions.

Now, while this exercise has given us interesting insights from an academic perspective, it doesn't ultimately help us achieve our objective. Given that SVM works best with two classes, and that our data might be *too* sparse for meaningful results, we might conclude that our data might be better suited to another model.

### Expanding our Analysis
We can include additional models and algorithms, which RTextTools makes very easy to execute. The following code divides our 18,898 records into four blocks of 4,000 records, with the remaining records used for testing. Each block is trained and classified, and 5-fold cross-validation is performed to determine accuracy. The default parameters for each algorithm are used.
```r
> for (idx in c(1:4000, 4001:8000, 8001:12000, 12001:16000)) {

>   data_container <- create_container(data_matrix, as.numeric(factor(data_rnd$genre)), trainSize=idx, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later
  
>   data_models <- train_models(data_container, algorithms = c("SVM", "MAXENT", "SLDA", "BOOSTING"))
>   data_results <- classify_models(data_container, data_models)
>   data_analytics <- create_analytics(data_container, data_results)
  
>   summary(data_analytics)
  
>   cross_SVM <- cross_validate(data_container, 5, "SVM")
>   cross_MAXENT <- cross_validate(data_container, 5, "MAXENT")
>   cross_SLDA <- cross_validate(data_container, 5, "SLDA")
>   cross_BOOSTING <- cross_validate(data_container, 5, "BOOSTING")

> }
```
If we collect the accuracy results from each cross-validation of each block of each model, we end up with the following graph:

![Cross-Validation Graph](/graphs/rtexttools-auto.png)

Interestingly, except for a single anomalous block in which *Maximum Entropy* performed admirably, only *Logitboost* (based on *AdaBoost*) produced consistent better-than-chance results.

### LogitBoost

Since we seemed to have some success with LogitBoost, let's look at its performance in more detail.
```r
> library("caTools")
> library("caret")

> boost_model <- LogitBoost(head(as.matrix(data_matrix_tfidf), 4000), head(as.numeric(factor(data_rnd$genre)), 4000), nIter=10)
```
Here we instruct R to create our LogitBoost model on the first 4000 records of our weighed data. Typically, the higher the number of iterations, *nIter*, the more accurate the model will be. Next, we can run the prediction and see how our confidence levels:
```r
> boost_scores <- predict(boost_model, tail(as.matrix(data_matrix_tfidf), 1000), nIter=10)
> boost_scores_prob <- predict(boost_model, tail(as.matrix(data_matrix_tfidf), 1000), nIter=10, type = "raw")
> t(cbind(boost_scores, round(boost_scores_prob, 4))[1:5,])
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
16           0.0180 0.0180 0.0180  0.0025 0.0025
21           0.0180 0.0180 0.0180  0.0180 0.0180
22           0.0025 0.0025 0.0025  0.0025 0.0025
23           0.1192 0.1192 0.1192  0.1192 0.1192
24           0.0025 0.0025 0.0025  0.0025 0.0025
25           0.0025 0.0180 0.0180  0.0180 0.0180
```
Note that I've transposed the results of this table using the *t* function for legibility, so that the documents are now the columns and the classes are now the rows. With that in mind, what this table allows us to see is the confidence, according to LogitBoost, of the first five documents falling into each of our 24 classes.

Differing levels of confidence can be found. For example, LogitBoost has 50% confidence of document 4 belonging to class 17, the highest result for that document. On the other hand, there's 50% confidence that document 3 belongs to either 10 or 17. The remaining documents have relatively poor confidence across the board.

We can gauge the overall accuracy of our model by visualizing a confusion matrix:
```r
> table(boost_scores, tail(as.numeric(factor(data_rnd$genre)), 1000))
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
Here we can see a number of true positives, particularly for classes 5, 7, 8, 14, 19 and 25, as well as a high number of incorrect classifications spread across every class. Ultimately, only 99 of 299 classified documents were correct. Let's increase the iterations to 20 and try again:
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
Here we see better representation among the classes, with more documents being classified, but our overall result of 97 of 342 true positives is still lacking. Additional work would therefore be needed to increase the accuracy of this model before we could it reliably. At this stage, we could also plot the area under the curve using R's *ROCR* library, but as it is limited to binary classification, we will skip this step.

Let's look back on our automated RTextTools model, since it produced favourable results:
```r
> data_container_tfidf <- create_container(data_matrix_tfidf, as.numeric(factor(data_rnd$genre)), trainSize = 1:4000, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later
> data_model <- train_model(data_container_tfidf, algorithm = "BOOSTING")
> data_results <- classify_model(data_container_tfidf, data_model)
> data_analytics <- create_analytics(data_container_tfidf, data_results)
```
In examining *data_model*, we can actually see that RTextTools has opted for 100 iterations. So let's map the confusion matrix again:
```r
> table(data_results$LOGITBOOST_LABEL, tail(as.numeric(factor(data_rnd$genre)), 2898))
```
```text
       1   2   3   4   5   6   7   8   9  10  11  13  14  15  16  17  18  19  20  21  22  23  24  25
  1   13   3   3   0  24  10   6  29   8   5   0   1   6   3   5   5   0   8   4   3   1  22   5   5
  10   3   3   0   0   4   0   0   5   0   1   0   0   3   0   0   0   0   3   0   1   0   4   0   1
  11   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0   0
  13   0   0   0   1   0   0   0   0   0   0   0   0   0   0   0   0   0   0   1   0   0   0   0   0
  14   2   3   2   0  16   6   2  12   0   2   0   1  32   0   1   6   0   2   2   1   0  22   0   0
  15   1   1   0   1   6   0   1   4   1   0   0   0   1   5   2   1   0   3   1   5   0   2   0   0
  16   2   1   1   0   8   0   1   6   4   0   0   0   2   1   3   1   0   3   0   3   0   1   0   2
  17   1   0   3   0   1   3   1   8   1   1   0   0   9   0   1   3   0   3   2   0   0   8   0   1
  19   3   6   0   0  10   4   1  23   0   2   0   3   0   1   4   2   0  18   2   3   0   2   0   0
  2   21  20   7   3  26   9  11  33   6   2   0   2   8   2   6   3   0   9   8   9   0  17   3   7
  20   3   2   1   0   4   0   2   3   2   1   0   0   4   0   0   0   0   0  12   2   0   3   0   0
  21   1   0   6   0  12   0   6   4   5   0   1   1   2   2   2   0   0   3   3  10   1   1   1   1
  22   0   0   1   1   2   0   1   1   1   0   0   0   2   0   0   0   0   0   0   0   4   0   0   0
  23  10   3   2   1  13   9   4  22   3   2   1   1  19   0   0   5   0   0   7   6   0  18   4   4
  24   3   1   0   0   1   0   0   2   0   0   0   0   0   0   0   0   0   0   0   0   0   1   2   0
  25   2   1   1   0   3   0   0   6   2   0   0   0   1   0   0   0   0   1   1   1   0   1   1  18
  3    2   2   9   0  20   4   3  10  12   2   0   1   3   0   3   1   0   2   1  16   0   3   0   0
  4    0   1   0   1   1   0   1   7   1   0   0   0   0   0   0   0   0   1   0   0   1   0   0   0
  5   32  10  15   9 228  24  24 197  35  11   5   5  41  14  15   8   0  78  17  42   9  44   1  18
  6   13   7   3   2  30  23   2  54   1   0   2   2  11   3   0   7   0   9   2   3   0  27   0  13
  7    2   1   0   8  10   3  58  25   1   1   0   6  11   7   0   1   4   3   2   5   0   6   4   0
  8   18  16   2  11  78  18  15 140   8   5   6   5  18   4   5   6   0  37  11  10   4  30   7  17
  9    2   1   4   0   7   2   1   5   2   0   0   0   6   1   0   0   0   1   1   6   0   1   0   0
```
Ultimately, our automated model produced 620 true positives out of 2898.

### Conclusion

With reduced sparsity and enough iterations, it should be possible to produce a model with a reasonable level of accuracy. This model can then be applied to predict the class of untrained data. That is, we can submit new plots and determine what genre our model thinks they would fall into. However, further experimentation is needed.

### Further Research

The following tasks may be worth pursuing to further improve results:

* Manipulate sparsity levels and determine affect on weighted matrices.
* Attempt different block sizes and determine affect on results.
* Test remaining models supported by RTextTools: Elastic net Regularization *(GLMNet)*, Bootstrap Aggregation *(Bagging)*, Random Forest *(RF)*, Neural Network *(NNET)*, and Trees *(TREE)*.
* Figure out to draw ROC curve for non-binary classification.
* Investigate e1071's [tune](http://www.inside-r.org/packages/cran/e1071/docs/tune) function.

### Reference R Library Manuals

* [RTextTools manual](https://cran.r-project.org/web/packages/RTextTools/RTextTools.pdf)
* [e1071 manual](https://cran.r-project.org/web/packages/e1071/e1071.pdf)
* [caTools manual](https://cran.r-project.org/web/packages/caTools/caTools.pdf)
* [caret manual](https://cran.r-project.org/web/packages/caret/caret.pdf)
* [SparseM manual](https://cran.r-project.org/web/packages/SparseM/SparseM.pdf)