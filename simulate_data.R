# ================= ============== simulated_data ===============================
# MCAR: Missing values were introduced by randomly masking non-zero entries, ensuring independence from observed or unobserved variables.
#	MAR: Missingness was biased toward rare cell subgroups (group 4 and group 5), making it dependent on cell subpopulation metadata.
#	MNAR: Missingness was concentrated in low-expression regions, linking it to the gene's own expression level, which is a common characteristic of technical dropouts in scRNA-seq.



# ------------------   The first section (MCAR)   -------------------------------
library(splatter)
set.seed(10100)
params <- newSplatParams()
params <- setParam(params, "nGenes", 4000)
params <- setParam(params, "batchCells", 2000)
params <- setParam(params, "dropout.type", "none")
group_probs <- c(0.3, 0.25, 0.2, 0.15, 0.1)
sim_nodrop <- splatSimulate(params_nodrop, method = "groups", group.prob = group_probs)
group_info <- colData(sim_dropout)$Group
write.csv(data.frame(Cell = colnames(sim_dropout), Group = group_info), 
          file = "cell_groups.csv", row.names = FALSE)
counts_nodrop <- as.data.frame(as.matrix(counts(sim_nodrop)))
write.csv(counts_nodrop, file = "simulated_counts_no_dropout.csv", quote = FALSE)





# ------------------   The second section (MAR)   -------------------------------
# Dropout parameter settings for different missing levels:
#
# Missing Level    dropout.mid                    dropout.shape
# ----------------------------------------------------------------
# 20%             c(0.3, 0.3, 0.3, 1.4, 1.4)       c(-1.0, -1.0, -1.0, -1.0, -1.0)
# 40%             c(1.5, 1.5, 1.5, 7.0, 7.0)       c(-1.0, -1.0, -1.0, -1.0, -1.0)
# 60%             c(4.0, 4.0, 4.0, 15.0, 15.0)     c(-1.0, -1.0, -1.0, -1.0, -1.0)
# ------------------------------------------------------------------------------
# Load the splatter library
library(splatter)
# Set seed for reproducibility
set.seed(10100)
params <- newSplatParams()
# Set number of genes and total number of cells
params <- setParam(params, "nGenes", 4000)
params <- setParam(params, "batchCells", 2000)
# Set dropout type to 'group', enabling different dropout patterns across groups
params <- setParam(params, "dropout.type", "group")
# Set dropout midpoint (dropout.mid) for each group
# These values control how strongly dropout is applied in each group
# Current setting corresponds to approximately 60% missing level
params <- setParam(params, "dropout.mid", c(4.0, 4.0, 4.0, 15.0, 15.0))
# Set dropout shape parameter (dropout.shape) for each group
# Negative values simulate realistic dropout curves (logistic function)
params <- setParam(params, "dropout.shape", c(-1.0, -1.0, -1.0, -1.0, -1.0))
# Define group probabilities for cells (five groups in total)
group_probs <- c(0.3, 0.25, 0.2, 0.15, 0.1)
# Simulate scRNA-seq data with dropout using the defined group probabilities
sim_dropout <- splatSimulate(params, method = "groups", group.prob = group_probs)
# Extract the count matrix and save it as a CSV file
counts_dropout <- as.data.frame(as.matrix(counts(sim_dropout)))
write.csv(counts_dropout, file = "simulated_counts_with_dropout.csv", quote = FALSE)
# Extract cell group information and save as a CSV file
group_info <- colData(sim_dropout)$Group
write.csv(data.frame(Cell = colnames(sim_dropout), Group = group_info), 
          file = "cell_groups.csv", row.names = FALSE)





# ------------------   The third section (MNAR)   -------------------------------
# Dropout parameter settings for different missing levels:
#
# Missing Level    dropout.mid    dropout.shape
# ---------------------------------------------
# 20%              0.6            -1.0
# 40%              2.2            -1.0
# 60%              4.2            -1.0
# ---------------------------------------------
#--------------------------------------------------------------------------------
# Load the splatter package
library(splatter)
# Set seed for reproducibility
set.seed(10100)
params <- newSplatParams()
# Set number of genes and cells
params <- setParam(params, "nGenes", 4000)
params <- setParam(params, "batchCells", 2000)
# Set dropout type to 'experiment' to simulate global dropout dependent on expression
# This models MNAR: missingness depends on gene expression level
params <- setParam(params, "dropout.type", "experiment")
# Set dropout midpoint: higher values increase dropout in low-expression regions
# Current setting corresponds to approximately 40% missingness
params <- setParam(params, "dropout.mid", 2.2)
# Set dropout shape
params <- setParam(params, "dropout.shape", -1.0)
# Define group probabilities for assigning cells to five groups
group_probs <- c(0.3, 0.25, 0.2, 0.15, 0.1)
# Simulate data with MNAR-style dropout
sim_dropout <- splatSimulate(params, method = "groups", group.prob = group_probs)
# Extract the simulated counts and save to CSV
counts_dropout <- as.data.frame(as.matrix(counts(sim_dropout)))
write.csv(counts_dropout, file = "simulated_counts_with_dropout.csv", quote = FALSE)
# Save cell group information
group_info <- colData(sim_dropout)$Group
write.csv(data.frame(Cell = colnames(sim_dropout), Group = group_info), 
          file = "cell_groups.csv", row.names = FALSE)
# Extract non-dropout counts
params_nodrop <- setParam(params, "dropout.type", "none")
sim_nodrop <- splatSimulate(params_nodrop, method = "groups", group.prob = group_probs)
counts_nodrop <- as.data.frame(as.matrix(counts(sim_nodrop)))
write.csv(counts_nodrop, file = "simulated_counts_no_dropout.csv", quote = FALSE)
# Missing rate of the simulated dataset
total_zeros <- sum(counts(sim_dropout) == 0)
total_bio_zeros <- sum(counts(sim_nodrop) == 0)
tech_zero_ratio_total <- (total_zeros - total_bio_zeros) / length(counts(sim_dropout))
print(paste("Proportion of missing zeros:", 
            signif(tech_zero_ratio_total * 100, 8), "%"))
print(paste("Number of missing zeros:", total_zeros - total_bio_zeros, 
            "/", length(counts(sim_dropout))))

