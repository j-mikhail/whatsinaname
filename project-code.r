# Step 1 - Clean data
# --------------------------------

# Data is available from: ftp://ftp.fu-berlin.de/pub/misc/movies/database/

# Data must be properly tab-delimited. See readme.md for more information.


# Step 2 - Import data
# --------------------------------

# Note: quotes = NULL must be used or apostrophes in titles disrupt record processing

data_ratings <- read.delim("ratings.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("rcount","rating","title"))
data_languages <- read.delim("language.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","language","variant"))
data_countries <- read.delim("countries.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","country"))
data_running_times <- read.delim("running-times.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","time","note"))
data_genres <- read.delim("genres.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","genre"))
data_plot <- read.delim("plot.tabbed.txt", header = FALSE, sep = "\t", quote=NULL, col.names = c("title","plot"))


# Step 3 - Crosslink data
# --------------------------------

# Note: default of all = FALSE cause inner join behavior

data_all <- data.frame(data_ratings["title"], data_ratings["rating"], data_ratings["rcount"])
data_all <- merge(data_all, data_running_times, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_languages, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_countries, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_genres, by.x = "title", by.y = "title")
data_all <- merge(data_all, data_plot, by.x = "title", by.y = "title")


# Step 4 - Filter data
# --------------------------------

data_all <- subset(data_all, !duplicated(title)) # Single genre per movie
data_all <- subset(data_all, language == "English") # English only
data_all <- subset(data_all, country == "USA") # USA only
data_all <- subset(data_all, !grepl("(TV)|(V)|(VG)", title)) # No TV shows, straight-to-videos or video games
data_all <- subset(data_all, !grepl("Adult|Erotica", genre)) # No porn
data_all <- subset(data_all, note == "") # No regional edits
data_all <- subset(data_all, rcount >= 100) # Only movies with at least 100 reviews

# OPTIONAL: Fix running times and convert to numeric, if to be used for classification
# data_all <- transform(data_all, time = as.numeric(gsub("(\\D+)(.*)", "\\2", time)))


# Step 5 - Sample data
# --------------------------------

data_rnd <- data_all[sample(nrow(data_all)),] # Randomize records
write.csv(data_rnd, "Clean-Random-Data.csv") # Save randomized records for reproducibility


# Step 6 - Scoring & Feature Selection
# --------------------------------

library("tm") # Required for TD-IDF
library("RTextTools") # Required for training

data_matrix_tf <- create_matrix(data_rnd$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTf) # Create matrix
data_matrix_tfidf <- create_matrix(data_rnd$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTfIdf) # Create matrix
data_container_tfidf <- create_container(data_matrix_tfidf, as.numeric(factor(data_rnd$genre)), trainSize=1:4000, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later

image(as.matrix.csr(data_container_tfidf@training_matrix)) # Map sparse matrix


# Step 7 - SVM Performance
# --------------------------------

# Linear Kernel

train_SVM_linear <- train_model(data_container_tfidf, "SVM", kernel = "linear") # Train
class_SVM_linear <- classify_model(data_container_tfidf, train_SVM_linear) # Test
anals_SVM_linear <- create_analytics(data_container_tfidf, class_SVM_linear) # Analytics

summary(anals_SVM_linear)
table(anals_SVM_linear@document_summary$SVM_LABEL, anals_SVM_linear@document_summary$MANUAL_CODE) # Results summary
table(anals_SVM_linear@document_summary$SVM_LABEL == anals_SVM_linear@document_summary$MANUAL_CODE) # Confusion Matrix

# Radial/Gaussian Kernel

train_SVM_radial <- train_model(data_container_tfidf, "SVM", kernel = "radial")
class_SVM_radial <- classify_model(data_container_tfidf, train_SVM_radial)
anals_SVM_radial <- create_analytics(data_container_tfidf, class_SVM_radial)

summary(anals_SVM_radial)
table(anals_SVM_radial@document_summary$SVM_LABEL, anals_SVM_radial@document_summary$MANUAL_CODE) # Results summary
table(anals_SVM_radial@document_summary$SVM_LABEL == anals_SVM_radial@document_summary$MANUAL_CODE) # Confusion Matrix

# Heatmaps

unfactor <- function(obj) {
  unfactor <- as.numeric(levels(obj)[as.integer(obj)])
}

heatmap(table(unfactor(anals_SVM_linear@document_summary$SVM_LABEL), anals_SVM_linear@document_summary$MANUAL_CODE)[length(unique(unfactor(anals_SVM_linear@document_summary$SVM_LABEL))):1,], Rowv = NA, Colv = NA, col = heat.colors(256))
heatmap(table(unfactor(anals_SVM_radial@document_summary$SVM_LABEL), anals_SVM_radial@document_summary$MANUAL_CODE)[length(unique(unfactor(anals_SVM_radial@document_summary$SVM_LABEL))):1,], Rowv = NA, Colv = NA, col = heat.colors(256))


# Step 8 - Class Balancing
# --------------------------------

# Histogram of genre distribution

data_block <- head(data_rnd$genre, 4000)
par(xaxt="n") # Turn off horizontal axis labels
plot(data_block, type="h", main = "Distribution of genres in first block") # Draw plot
par(xaxt="s") # Turn back on horizontal axis labels
axis(1, at=seq(par("xaxp")[1], par("xaxp")[2], by=(par("xaxp")[2]-par("xaxp")[1])/(length(unique(data_block))+1)), labels = FALSE) # Draw horizontal ticks
incr <- (par("xaxp")[2]-par("xaxp")[1])/(length(unique(data_block))+1) # Calculate distance between bars
text(x = seq(par("xaxp")[1]+incr, par("xaxp")[2], by=incr), y = -20, labels = sort(unique(data_rnd$genre)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8) # Write horizontal labels

# Create balanced data set with max. 250 records per class, and no bum classes

data_balanced <- NULL
for (g in 1:length(levels(data_rnd$genre))) {
  data_balanced <- rbind(data_balanced, head(subset(data_rnd, genre == levels(data_rnd$genre)[g]), 250))
}
data_balanced <- subset(data_balanced, !grepl("Game-Show|News", genre)) # Remove empty or near-empty classes
data_balanced$genre <- factor(data_balanced$genre) # Re-factorize
data_balanced <- data_balanced[sample(nrow(data_balanced)),] # Randomize

# Histogram of balanced class distribution

par(xaxt="n") # Turn off horizontal axis labels
plot(data_balanced$genre, type="h", main = "Distribution of genres in balanced set") # Draw plot
par(xaxt="s") # Turn back on horizontal axis labels
incr <- par("usr")[2]/(length(levels(data_balanced$genre))+1) # Calculate distance between bars
axis(1, at=seq(0, par("usr")[2]-incr, by=incr), labels = FALSE) # Draw horizontal ticks
text(x = seq(0+incr, par("usr")[2]-incr, by=incr), y = -20, labels = sort(levels(data_balanced$genre)), srt = 90, pos = 2, xpd = TRUE, cex = 0.8) # Write horizontal labels


# Step 9 - Balanced SVM Performance
# --------------------------------

# Create new balanced matrix and container

data_balanced_matrix_tfidf <- create_matrix(data_balanced$plot, language = "english", stemWords = TRUE, removeStopwords = TRUE, removeNumbers = TRUE, removePunctuation = TRUE, removeSparseTerms = 0.998, weighting = weightTfIdf) # Create matrix
data_balanced_container_tfidf <- create_container(data_balanced_matrix_tfidf, as.numeric(factor(data_balanced$genre)), trainSize=1:4000, testSize = 4001:5365, virgin = FALSE) # Factorization of class is necessary or analysis will fail later

# Linear Kernel

train_SVM_linear <- train_model(data_balanced_container_tfidf, "SVM", kernel = "linear")
class_SVM_linear <- classify_model(data_balanced_container_tfidf, train_SVM_linear)
anals_SVM_linear <- create_analytics(data_balanced_container_tfidf, class_SVM_linear)

summary(anals_SVM_linear)
table(unfactor(anals_SVM_linear@document_summary$SVM_LABEL), anals_SVM_linear@document_summary$MANUAL_CODE) # Confusion Matrix
table(anals_SVM_linear@document_summary$SVM_LABEL == anals_SVM_linear@document_summary$MANUAL_CODE) # Accuracy

# Heatmap

heatmap(table(unfactor(anals_SVM_linear@document_summary$SVM_LABEL), anals_SVM_linear@document_summary$MANUAL_CODE)[length(unique(unfactor(anals_SVM_linear@document_summary$SVM_LABEL))):1,], Rowv = NA, Colv = NA, col = heat.colors(256))


# Step 10 - LogitBoost Performance
# --------------------------------

# Using caTools

library("caTools")
library("caret")

boost_model <- LogitBoost(head(as.matrix(data_matrix_tfidf), 4000), head(as.numeric(factor(data_rnd$genre)), 4000), nIter=20) # Train, 20 iterations
boost_scores <- predict(boost_model, tail(as.matrix(data_matrix_tfidf), 1000)) # Test
boost_scores_prob <- predict(boost_model, tail(as.matrix(data_matrix_tfidf), 1000), type = "raw") # Analytics

t(cbind(boost_scores, round(boost_scores_prob, 4))[1:5,]) # Visualize classification probabilities for first 5 documents
table(boost_scores, tail(as.numeric(factor(data_rnd$genre)), 1000)) # Confusion matrix

# Using RTextTools

data_balanced_model <- train_model(data_balanced_container_tfidf, algorithm = "BOOSTING")
data_balanced_results <- classify_model(data_balanced_container_tfidf, data_balanced_model)
data_balanced_analytics <- create_analytics(data_balanced_container_tfidf, data_balanced_results)

table(unfactor(data_balanced_analytics@document_summary$LOGITBOOST_LABEL), data_balanced_analytics@document_summary$MANUAL_CODE) # Confusion Matrix
table(data_balanced_analytics@document_summary$LOGITBOOST_LABEL == data_balanced_analytics@document_summary$MANUAL_CODE) # Accuracy


# Step 11 - Additional Algorithm Performance
# --------------------------------

do_analysis <- function(idx, algs, cf = 0) {
  
  data_container_tfidf <- create_container(data_matrix_tfidf, as.numeric(factor(data_rnd$genre)), trainSize=idx, testSize = 16001:18898, virgin = FALSE) # Factorization of class is necessary or analysis will fail later
  
  for (alg in algs) {
    
    print(paste0("Training ", alg, "..."))
    data_model <- train_model(data_container_tfidf, algorithm = alg)
    print(paste0("Classifying ", alg, "..."))
    data_results <- classify_model(data_container_tfidf, data_model)
    print(paste0("Creating analytics for ", alg, "..."))
    data_analytics <- create_analytics(data_container_tfidf, data_results)
    
    print(paste0("Saving summary data for ", alg, "..."))
    print(summary(data_analytics))
    
    # Save data
    write.csv(data_analytics@document_summary, paste0(alg, "-Sum-Doc-", idx[1], "-", idx[length(idx)], ".csv"))
    write.csv(data_analytics@algorithm_summary, paste0(alg, "-Sum-Alg-", idx[1], "-", idx[length(idx)], ".csv"))
    write.csv(data_analytics@ensemble_summary, paste0(alg, "-Sum-Ens-", idx[1], "-", idx[length(idx)], ".csv"))
    write.csv(data_analytics@label_summary, paste0(alg, "-Sum-Top-", idx[1], "-", idx[length(idx)], ".csv"))
    
    if (cf > 0) {
      # Cross validation
      print(paste0(cf, "-fold cross-validating ", alg, "..."))
      cross_data <- cross_validate(data_container_tfidf, cf, alg)
      write.csv(cross_data, paste0(alg, "-Cross-", idx[1], "-", idx[length(idx)], ".csv"))
    }
    
    gc()
    
  }
  
}

# Sequential

do_analysis(c(1:4000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))
do_analysis(c(4001:8000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))
do_analysis(c(8001:12000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))
do_analysis(c(12001:16000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))

# Cumulative

do_analysis(c(1:4000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))
do_analysis(c(1:8000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))
do_analysis(c(1:12000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))
do_analysis(c(1:16000), c("SVM", "MAXENT", "SLDA", "BOOSTING"))

# Graph example

par(xaxt="n") # Turn off horizontal axis labels
plot(type = "o", x = c(1:4), y = subset(auto_results, V2 == unique(auto_results$V2)[1])$V3, ylim=c(0.2, 0.45), col=c("red", "blue", "green", "orange")[1], ylab = "Accuracy", xlab="Sequential Documents", main = "Sequential Classification Accuracy", pch = 19)
for (i in 2:length(unique(auto_results$V2))) {
  points(type = "o", x = c(1:4), y = subset(auto_results, V2 == unique(auto_results$V2)[i])$V3, col=c("red", "blue", "green", "orange")[i], pch = 19)
}
par(xaxt="s") 
axis(1, 1:4, unique(auto_results$V1)) # Draw horizontal ticks and labels
legend("topright", legend = unique(auto_results$V2), col=c("red", "blue", "green", "orange"), text.col=c("red", "blue", "green", "orange"), cex=0.8, pch=19)

# 5-fold cross-validation and plot example

do_analysis(c(1:4000), c("SVM", "MAXENT", "SLDA", "BOOSTING", "GLMNET", "NNET", "RF", "TREE"), 5)
plot(x = cf_data$Algo, y = cf_data$Accuracy, type="o", col=as.numeric(factor(unique(cf_data$Algo))), xlab = "Algorithm", ylab = "Accuracy", main = "5-Fold Cross-Validation Performance\n4000 documents")