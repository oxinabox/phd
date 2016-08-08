### Model-baseed clustering of time series data
### Prac 2: Markov random fields
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
# To illustrate how Markov random fields can be used to 
# recover smooth signals from noisy signals. We can use a
# simulation study.

# Simulate data from the simulation paradigm from the
# lecture.
simu <- matrix(NA,100,100)
for (ii in 1:100) {
  for (jj in 1:100) {
    if (floor((ii-1/2)/20) %in% c(0,2,4)) {
      if (floor((jj-1/2)/20) %in% c(0,2,4)) {
        simu[ii,jj] <- 1
      } else {
        simu[ii,jj] <- 2
      }
    } else {
      if (floor((jj-1/2)/20) %in% c(1,3)) {
        simu[ii,jj] <- 3
      } else {
        simu[ii,jj] <- 4
      }
    }
  }
}

# Plot an image of the simulation paradigm.
image.plot(simu)

# Corrupt 20% of the simulated data in order to artifically
# introduce noise.
simu_corr <- simu
x_co <- sample(1:100,2000,replace=T)
y_co <- sample(1:100,2000,replace=T)
for (ii in 1:2000) {
  simu_corr[x_co[ii],y_co[ii]] <- sample(c(1,2,3,4),1)
}

# Define a function that obtains the Moore neighborhood
# of distance 'dist' for any coordinate (x_co,y_co)
Moore_nbh <- function(input,dist,x_co,y_co) {
  C_xy <- input[x_co,y_co]
  output <- input[max(1,x_co-dist):min(dim(input)[1],x_co+dist),
                   max(1,y_co-dist):min(dim(input)[2],y_co+dist)]
  output <- c(output)
  output <- output[-which(output==C_xy)[1]]
  output
}

# Define a function that obtains the von Neumann
# neighborhood of distance 'dist' for any coordinate
# (x_co,y_co).
vN_nbh <- function(input,dist,x_co,y_co) {
  C_xy <- input[x_co,y_co]
  output <- c()
  count <- 0
  list_x <- max(1,x_co-dist):min(dim(input)[1],x_co+dist)
  list_y <- max(1,y_co-dist):min(dim(input)[2],y_co+dist)
  for (aa in list_x) {
    for (bb in list_y) {
      if (abs(aa-x_co) + abs(bb-y_co) <= dist) {
        count <- count + 1
        output[count] <- input[aa,bb]
      }
    }
  }
  output <- output[-which(output==C_xy)[1]]
  output
}

# Create a variable for the function evaluates eta_fun and
# put into the function, the average count from each of the
# 4 classes, within the Moore neighborhood of distance 1
# around each coordinate (x,y).
count <- 0
dist <- 1
eta_fun <- matrix(NA,100*100,4)
for (x_co in 1:100) {
  for (y_co in 1:100) {
    count <- count + 1
    eta_fun[count,1] <- mean(Moore_nbh(simu_corr,dist,x_co,y_co)==1,na.rm=T) 
    eta_fun[count,2] <- mean(Moore_nbh(simu_corr,dist,x_co,y_co)==2,na.rm=T) 
    eta_fun[count,3] <- mean(Moore_nbh(simu_corr,dist,x_co,y_co)==3,na.rm=T) 
    eta_fun[count,4] <- mean(Moore_nbh(simu_corr,dist,x_co,y_co)==4,na.rm=T) 
  }
}

# Construct an MRF from the image and neighbourhood function
# eta_fun using the multinomial regression function multinom.
noise_cat <- c(simu_corr)
MRF_1 <- multinom(noise_cat ~ eta_fun[,1]+eta_fun[,2]+
                    eta_fun[,3]+eta_fun[,4],maxit=1000) 

# Obtain the parameter estimates from the fitted MRF model.
summary(MRF_1)

# Compute the PLIC value of the fitted MRF model.
BIC(MRF_1)

# Plot an image of the obtained estimated signals from the
# MRF model.
smooth_im_1 <- matrix(as.numeric(predict(MRF_1)),100,100)
image.plot(smooth_im_1)

# Create a variable for the function evaluates eta_fun and
# put into the function, the average count from each of the
# 4 classes, within the vN neighborhood of distance 1
# around each coordinate (x,y).
count <- 0
dist <- 1
eta_fun <- matrix(NA,100*100,4)
for (x_co in 1:100) {
  for (y_co in 1:100) {
    count <- count + 1
    eta_fun[count,1] <- mean(vN_nbh(simu_corr,dist,x_co,y_co)==1,na.rm=T) 
    eta_fun[count,2] <- mean(vN_nbh(simu_corr,dist,x_co,y_co)==2,na.rm=T) 
    eta_fun[count,3] <- mean(vN_nbh(simu_corr,dist,x_co,y_co)==3,na.rm=T) 
    eta_fun[count,4] <- mean(vN_nbh(simu_corr,dist,x_co,y_co)==4,na.rm=T) 
  }
}

# Construct an MRF from the image and neighbourhood function
# eta_fun using the multinomial regression function multinom.
noise_cat <- c(simu_corr)
MRF_2 <- multinom(noise_cat ~ eta_fun[,1]+eta_fun[,2]+
                    eta_fun[,3]+eta_fun[,4],maxit=1000) 

# Obtain the parameter estimates from the fitted MRF model.
summary(MRF_2)

# Compute the PLIC value of the fitted MRF model.
BIC(MRF_2)

# Plot an image of the obtained estimated signals from the
# MRF model.
smooth_im_2 <- matrix(as.numeric(predict(MRF_2)),100,100)
image.plot(smooth_im_2)

# Construct an MRF from the image and neighbourhood function
# eta_fun and interactions between the functions using the
# multinomial regression function multinom.
noise_cat <- c(simu_corr)
MRF_3 <- multinom(noise_cat ~ (eta_fun[,1]+eta_fun[,2]+
                    eta_fun[,3]+eta_fun[,4])^2,maxit=1000) 
BIC(MRF_3)
smooth_im_3 <- matrix(as.numeric(predict(MRF_3)),100,100)
image.plot(smooth_im_3)

#* Create eta_fun functions of Moore and vN of distances
#* dist = 2, 5, and 10 and compare the plotted images and
#* PLIC values with those obtained from distance 1, with
#* and without interactions.

#* Try defining more complex functions for eta_fun and see
#* if it makes a difference. For example, try using the
#* variances, or polynomial terms.

#* Redefine the Moore and vN neighbourhoods to be inclusive
#* of the coordinate (x,y) and see if it makes a difference
#* to the obtained signal recovery.

#* Repeat the exercise with 50% randomly corrupted data
#* instead of 20%.


# Exercise 2 ------------------------------------------
# We now link together the MRF and GMM clustering from 
# Prac 1. Recall that the data from Prac 1 is from
# http://www.bic.mni.mcgill.ca/ServicesAtlases/ICBM152NLin2009.
# Of the available atlases, we are only interested in 
# the mni_icbm152_t2_tal_nlin_sym_09a.nii file. Copy this
# file into your R working directory.

# Using the oro.nifti package, we can read in the file.
brain_image <- readNIfTI('mni_icbm152_t2_tal_nlin_sym_09a.nii')

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

# Use the mergenormals function from the fpc package to
# create 5 clusters via merging, using the 'demp' method.
merge_1 <- mergenormals(c(z_slice),GMM_Brain_1,
                        method='demp',numberstop = 5)

# Obtain and plot the 5 cluster signal upon merging.
cluster_mat_1 <- matrix(merge_1$clustering,197,233)
image.plot(cluster_mat_1)

# Create a variable for the function evaluates eta_fun and
# put into the function, the average count from each of the
# 5 classes, within the Moore neighborhood of distance 1
# around each coordinate (x,y).
count <- 0
dist <- 1
eta_fun <- matrix(NA,197*233,5)
for (y_co in 1:233) {
  for (x_co in 1:197) {
    count <- count + 1
    eta_fun[count,1] <- mean(Moore_nbh(cluster_mat_1,dist,x_co,y_co)==1,na.rm=T) 
    eta_fun[count,2] <- mean(Moore_nbh(cluster_mat_1,dist,x_co,y_co)==2,na.rm=T) 
    eta_fun[count,3] <- mean(Moore_nbh(cluster_mat_1,dist,x_co,y_co)==3,na.rm=T) 
    eta_fun[count,4] <- mean(Moore_nbh(cluster_mat_1,dist,x_co,y_co)==4,na.rm=T) 
    eta_fun[count,5] <- mean(Moore_nbh(cluster_mat_1,dist,x_co,y_co)==5,na.rm=T) 
  }
}

# Construct an MRF from the image and neighbourhood function
# eta_fun using the multinomial regression function multinom.
noise_cat <- c(cluster_mat_1)
MRF_4 <- multinom(noise_cat ~ eta_fun[,1]+eta_fun[,2]+
                    eta_fun[,3]+eta_fun[,4]+eta_fun[,5],
                  maxit=1000) 
BIC(MRF_4)
smooth_im_4 <- matrix(as.numeric(predict(MRF_4)),197,233,byrow=F)
image.plot(smooth_im_4)

# Create a variable for the function evaluates eta_fun and
# put into the function, the average count from each of the
# 5 classes, within the vN neighborhood of distance 1
# around each coordinate (x,y).
count <- 0
dist <- 1
eta_fun <- matrix(NA,197*233,5)
for (y_co in 1:233) {
  for (x_co in 1:197) {
    count <- count + 1
    eta_fun[count,1] <- mean(vN_nbh(cluster_mat_1,dist,x_co,y_co)==1,na.rm=T) 
    eta_fun[count,2] <- mean(vN_nbh(cluster_mat_1,dist,x_co,y_co)==2,na.rm=T) 
    eta_fun[count,3] <- mean(vN_nbh(cluster_mat_1,dist,x_co,y_co)==3,na.rm=T) 
    eta_fun[count,4] <- mean(vN_nbh(cluster_mat_1,dist,x_co,y_co)==4,na.rm=T) 
    eta_fun[count,5] <- mean(vN_nbh(cluster_mat_1,dist,x_co,y_co)==5,na.rm=T) 
  }
}

# Construct an MRF from the image and neighbourhood function
# eta_fun using the multinomial regression function multinom.
noise_cat <- c(cluster_mat_1)
MRF_5 <- multinom(noise_cat ~ eta_fun[,1]+eta_fun[,2]+
                    eta_fun[,3]+eta_fun[,4]+eta_fun[,5],
                  maxit=1000) 
BIC(MRF_5)
smooth_im_5 <- matrix(as.numeric(predict(MRF_5)),197,233,byrow=F)
image.plot(smooth_im_5)

# Construct an MRF from the image and neighbourhood function
# eta_fun and interactions between the functions using the
# multinomial regression function multinom.
noise_cat <- c(cluster_mat_1)
MRF_6 <- multinom(noise_cat ~ (eta_fun[,1]+eta_fun[,2]+
                    eta_fun[,3]+eta_fun[,4]+eta_fun[,5])^2,
                  maxit=1000) 
BIC(MRF_6)
smooth_im_6 <- matrix(as.numeric(predict(MRF_6)),197,233,byrow=F)
image.plot(smooth_im_6)

#* Create eta_fun functions of Moore and vN of distances
#* dist = 2, 3, and 5 and compare the plotted images and
#* PLIC values with those obtained from distance 1, with
#* and without interactions.

#* Try defining more complex functions for eta_fun and see
#* if it makes a difference. For example, try using the
#* variances, or polynomial terms.

#* Redefine the Moore and vN neighbourhoods to be inclusive
#* of the coordinate (x,y) and see if it makes a difference
#* to the obtained signal recovery.

#* Repeat the exercise with 50% randomly corrupted data
#* instead of 20%.

# We can corrupt the image randomly at 20% of the points.
brain_corr <- cluster_mat_1
x_co <- sample(1:197,10000,replace=T)
y_co <- sample(1:233,10000,replace=T)
for (i in 1:10000) {
  brain_corr[x_co[i],y_co[i]] <- sample(c(1,2,3,4,5),1)
}
image.plot(brain_corr)

#* Repeat the exercise using the corrupted brain image
#* instead.

#* Repeat the exercise using the 100th z-slice insteas of
#* the 94th.