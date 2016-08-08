### Model-baseed clustering of time series data
### Prac 1: Model-based clustering
### Hien Nguyen, University of Queensland
### 23/06/2016

## If you haven't already done so, download R from
## https://cran.r-project.org
## Download RStudio from https://www.rstudio.com
## Install the necessary packages via the command
install.packages('oro.nifti','fields','flexmix','mclust',
                 'nnet','fpc')


# Load necessary libraries for practicals
library(oro.nifti)
library(fields)
library(flexmix)
library(mclust)
library(nnet)
library(fpc)
library(MASS)

# Help for R functions can be obtained using ?
# For example, try: ?iris
# Alternatively, there is a comprehensive community of
# R users on the internet. Simply google any topic, and the
# letter R, and you will be likely to obtain help.


# Exercise 1 ------------------------------------------
# We start with the standard 'hello world' exercise to
# demonstrate the syntax of R.

# Print 'hello world' to the console.
print('hello world')

# Store hello world into a convenient variable.
hw <- 'hello world'

#*1 Now try printing your 'hello YOUR_NAME'.
greetme<-'hello Lyndon_White'
print(greetme)

#* Put 'hello YOUR_NAME' into a convenient variable and
#* print it.

# Exercise 2 ------------------------------------------
# R is mainly designed to be an interactive tool for data
# analysis, that also happens to be an excellent programming
# language. A core freature of R is its graphical rendering
# of data. We will demonstrate this with the iris data set.

# Load the iris data set.
data(iris)

# Display the iris data set.
print(iris)

# Plot the pairwise projections of the iris data set.
plot(iris[,1:4])

# Color the data by the subspecies.
plot(iris[,1:4],col=iris[,5])

# Use different colors.
plot(iris[,1:4],col=c('orange','violet','cyan')[iris[,5]])

# Use different symbols for each subspecies.
plot(iris[,1:4],col=iris[,5],pch=c(1,2,3)[iris[,5]])

# Plot only some of the features.
plot(iris[,1:2], col=iris[,5])

# Plot a histogram of the Sepal Length
hist(iris[,1])

# Rename the different elements of the histogram to be
# more informative.
hist(iris[,1],main='Histogram of Sepal Length',
     xlab='Length',ylab='Count')

# Plot a kernel density estimate of the Sepal Length.
plot(density(iris[,1]))

#* Plot the Petal Length and Petal Width with different
#* shapes and colors for each subspecies.

plot(iris[,c('Petal.Length', 'Petal.Width')], col=iris[,5])

#* Load the 'diabetes' data set
data('diabetes')
#* Plot the 'diabetes' data set with different colors and
#* shapes for each class.

plot(diabetes, col=diabetes[,'class'], pch=c(4,2,3)[diabetes[,"class"]])

#* Plot a histogram of the glucose levels, with informative
#* title and labels.
hist(diabetes[,'glucose'])
#,main='Histogram of Sepal Length',xlab='Length',ylab='Count')


#* Plot a kernel density estimate with informative title and
#* labels.
plot(density(diabetes[,'glucose']))


# Exercise 3 ------------------------------------------
# R natively offers the k-means algorithm for clustering,
# which is a very powerful solution, that is fast and
# efficient. We will demonstrate its use here.

# Obtain a k-means clustering of the iris data using
# k = 3 clusters.
k_means_1 <- kmeans(x = iris[,1:4], centers = 3)

# Print the output of the k-means clustering.
print(k_means_1)

# Plot the iris data coloured by the k-means.
plot(iris[,1:4],col=k_means_1$cluster)

# Repeat the clustering and plotting of the data.
k_means_2 <- kmeans(x = iris[,1:4], centers = 3)
plot(iris[,1:4],col=k_means_2$cluster)

# Perform the clustering with 20 restarts.
k_means_3 <- kmeans(x = iris[,1:4], centers = 3,
                    nstart = 200)
plot(iris[,1:4],col=k_means_3$cluster)

#* Perform the clustering using k = 2 instead of 3 clusters,
#* and plot the clustering with different colors for each
#* cluster.

k_means_4 <- kmeans(x = iris[,1:4], centers = 10,
                    nstart = 20)
plot(iris[,1:4],col=k_means_4$cluster)


#* Perform the clustering using k = 3 clusters, but only on
#* the Sepal Length and Sepal Width variables.

k_means_5 <- kmeans(x = iris[,c('Sepal.Length', 'Sepal.Width')], centers = 3,
                    nstart = 20)
plot(iris[,c('Sepal.Length', 'Sepal.Width')],col=k_means_5$cluster, pch=4)


#* Perform a clustering using k = 3 clusters on the diabetes
#* data and plot the clustering with different colors for
#* each cluster.

k_means_6 <- kmeans(x = diabetes[,c('glucose','insulin','sspg')], centers = 4,
                    nstart = 20)
plot(diabetes[,c('glucose','insulin','sspg')],col=k_means_6$cluster, pch=c(1,2,3)[diabetes[,'class']])


#* Perform a clustering using k = 3 clusters on only the
#* glucose and insulin variables of the diabetes data set.


k_means_7 <- kmeans(x = diabetes[,c('glucose','insulin')], centers = 4,
                    nstart = 20)
plot(diabetes[,c('glucose','insulin')],col=k_means_7$cluster,pch=c(1,2,3)[diabetes[,'class']])


# Exercise 4 ------------------------------------------
# To assess whether two sets of labels are coherent, we
# can use the adjusted-Rand index that is available from
# the mclust package.

# Compare the k = 3 clustering of the iris data against
# the true subspecies labels.
adjustedRandIndex(iris[,5],k_means_3$cluster)

# Compare the k = 4 clustering of the iris data against
# the true subspecies labels.
k_means_4 <- kmeans(x = iris[,1:4], centers = 3,
                    nstart = 20)
plot(iris[1:4],col=k_means_4$cluster)
adjustedRandIndex(iris[,5],k_means_4$cluster)

#* Compare the k = 3 clustering of the diabetes data against
#* the disease classes.

k_means_4 <- kmeans(x = diabetes[,2:4], centers = 3,
                    nstart = 20)
plot(diabetes[,2:4],col=k_means_4$cluster, pch=c(1,2,3)[diabetes[,1]])
adjustedRandIndex(diabetes[,1],k_means_4$cluster)


#* Compare the k = 2 clustering of the diabetes data against
#* the disease classes.

k_means_4 <- kmeans(x = diabetes[,2:4], centers = 3,
                    nstart = 20)
plot(diabetes[,2:4],col=k_means_4$cluster, pch=c(1,2,3)[diabetes[,1]])
adjustedRandIndex(diabetes[,1],k_means_4$cluster)



# Exercise 5 ------------------------------------------
# The mclust package provides provisions for clustering via
# Gaussian mixture models. It also automates the selection
# of these mixture models using the BIC rule.

# Cluster the iris data using a 3-component GMM
GMM_1 <- Mclust(iris[,1:4],G=3,modelNames = 'VVV')

# Obtain summary statistics regarding the clustering.
summary(GMM_1)

# Obtain the names of the different variables stored within
# the object GMM_1
attributes(GMM_1)

# Print the clustering of each observation, obtained in
# GMM_1.
print(GMM_1$classification)

# Plot the iris data coloured by the GMM clustering.
plot(iris[,1:4],col=GMM_1$classification, pch=c(1,2,3)[iris[,'Species']])

# Perform a GMM clustering of the iris data without
# declaring the number of clusters.
GMM_2 <- Mclust(iris[,1:4],modelNames = 'VVV')
summary(GMM_2)

# Plot the BIC values obtained from these clusterings
plot(GMM_2,what='BIC')

#* Cluster the diabetes data using a 3-component GMM
diaGMM3 <- Mclust(diabetes[,2:4], modelNames = 'VVV', G=3)
summary(diaGMM3)
#* Plot the diabetes data coloured by the GMM clustering.

#* Perform a GMM clustering of the diabetes data without
#* declaring the number of clusters.
diaGMM <- Mclust(diabetes[,2:4], modelNames = 'VVV')
summary(diaGMM)

#* Plot the BIC values obtained from these clusterings
plot(diaGMM,what='BIC')


#* By setting what = 'classification', 'uncertainty', or
#* 'density', we obtain 3 additional plots regarding the
#* GMM clustering. What do these plots do?
plot(diaGMM,what=c('classification','BIC','density','uncertainty'))


#* Try leaving out the modelNames = 'VVV' part of the 
#* Mclust call and explain what is happening.

diaGMM <- Mclust(diabetes[,2:4])
summary(diaGMM)


#* Use the adjusted-Rand index to inspect how coherent
#* the clustering is when compared to the true disease
#* classes.
adjustedRandIndex(diabetes[,1],diaGMM$classification)


# Exercise 6 ------------------------------------------
# Numerous neuroimaging data sets are freely available
# online. One such data set is the 2009a BIC MNI Brain Atlas
# that is available from:
# http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009.
# Of the available atlases, we are only interested in 
# the mni_icbm152_t2_tal_nlin_sym_09a.nii file. Copy this
# file into your R working directory.

# Using the oro.nifti package, we can read in the file.
brain_image <- readNIfTI(
  './mni_icbm152_nlin_sym_09a/mni_icbm152_t2_tal_nlin_sym_09a.nii')

# From the image, we only want a single z-slice, say 94.
z_slice <- brain_image[,,94]

# Plot the image of the 94th z-slice using the fields
# package.
image.plot(z_slice)

# Plot a histogram of the image intensities
hist(z_slice)

# Plot a kernel density estimate over the intensities
plot(density(z_slice))

# Cluster the intensities using a GMM without declaring the
# number of clusters.
GMM_Brain_1 <- Mclust(c(z_slice),G=1:20,modelNames = 'V')

# Get the dimensionality of the z_slice
dim(z_slice)

# Put the GMM clustering back into a matrix of the same
# size as the z_slice, and plot it.
cluster_mat_1 <- matrix(GMM_Brain_1$classification,197,233)
image.plot(cluster_mat_1)

# Use the mergenormals function from the fpc package to
# find the optimal number of clusters to merge together,
# using the 'demp' method.
merge_1 <- mergenormals(c(z_slice),GMM_Brain_1,
                        method='demp')
cluster_mat_2 <- matrix(merge_1$clustering,197,233)
image.plot(cluster_mat_2)

# Use the mergenormals function from the fpc package to
# create 5 clusters via merging, using the 'demp' method.
merge_2 <- mergenormals(c(z_slice),GMM_Brain_1,
                        method='demp',numberstop = 5)
cluster_mat_3 <- matrix(merge_2$clustering,197,233)
image.plot(cluster_mat_3)

#* Repeat the exercise for z-slice 100.