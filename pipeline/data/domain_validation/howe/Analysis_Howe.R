d <- read.csv('Data_Howe et al_Acknowledging Uncertainty.csv')

##### Loading relevant libraries and functions #####

library(psych); library(esc); library(DescTools); library(mediation); library(lsr); library(effsize)

partial.R2<-function(nested.lm, ref.lm){
  a <- anova(nested.lm)
  b <- anova(ref.lm)
  length.ref <- length(attributes(ref.lm$terms)$"dataClasses")
  length.nested <- length(attributes(nested.lm$terms)$"dataClasses")
  if(length.nested > length.ref) stop("Specify nested model first in arguements")
  if(length.ref - length.nested > 1) stop("Reference and nested model should only differ with repsect to the presence/absence of one predictor")
  SSE.wo <- tail(a$"Sum Sq", 1)
  SSE.with <- tail(b$"Sum Sq", 1)
  P.R2<-(SSE.wo-SSE.with)/SSE.wo
  P.R2
}

##### Creating the condition variables #####

d$condmean[d$Condition=='Fully Bounded'] <- 'fully bounded'
d$condmean[d$Condition=='Fully Bounded and Irreducible'] <- 'fully bounded'
d$condmean[d$Condition=='Partially Bounded'] <- 'partially bounded'
d$condmean[d$Condition=='Partially Bounded and Irreducible'] <- 'partially bounded'
d$condmean[d$Condition=='No Uncertainty'] <- 'no bounds'
d$condmean[d$Condition=='Irreducible'] <- 'no bounds'
d$condmean <- as.factor(d$condmean)

d$condstorm[d$Condition=='Fully Bounded'] <- 'no irreducible'
d$condstorm[d$Condition=='Fully Bounded and Irreducible'] <- 'irreducible'
d$condstorm[d$Condition=='Partially Bounded'] <- 'no irreducible'
d$condstorm[d$Condition=='Partially Bounded and Irreducible'] <- 'irreducible'
d$condstorm[d$Condition=='No Uncertainty'] <- 'no irreducible'
d$condstorm[d$Condition=='Irreducible'] <- 'irreducible'
d$condstorm <- as.factor(d$condstorm)

# Creating the message acceptance index

items <- list(items=c('SLReffects','STeffects','stormserious'))
results<-scoreItems(items, d, impute='none')
results
d$seriousness <- as.numeric(results$scores)

# Creating dummy codes for calculating the effect size of interactions

d$mean <- ifelse(d$condmean == 'no bounds', 1,0)
d$high <- ifelse(d$condmean == 'partially bounded', 1,0)
d$both <- ifelse(d$condmean == 'fully bounded', 1,0)

# Creating a variable combining comparison group so that it is both no bounded uncertainty and partially bounded uncertainty

d$condfully[d$condmean == 'no bounds'] <- 'none/partial'
d$condfully[d$condmean == 'partially bounded'] <- 'none/partial'
d$condfully[d$condmean == 'fully bounded'] <- 'z fully bounded'
d$condfully <- as.factor(d$condfully)

##### Removing participants who do not have a value for message acceptance because it could not be imputed #####
d <- subset(d, seriousness >= 0)

##### Setting contrasts for analysis #####

contrasts(d$educ) <- cbind(hsvsome=c(0,0,0,1),hsvcollgrad=c(1,0,0,0),hsvref=c(0,0,1,0))
contrasts(d$age) <- cbind(v1824=c(1,0,0,0,0,0),v2534=c(0,1,0,0,0,0),v3544=c(0,0,1,0,0,0),
                          v4554=c(0,0,0,1,0,0),v5564=c(0,0,0,0,1,0))
contrasts(d$income)=cbind(v3050=c(0,1,0,0,0),v5075=c(0,0,1,0,0),v75100=c(0,0,0,1,0),v100=c(1,0,0,0,0))
contrasts(d$region) <- cbind(vmw=c(1,0,0,0),vne=c(0,1,0,0),vs=c(0,0,1,0))
contrasts(d$partyID) <- cbind(vdem=c(1,0,0,0),vref=c(0,0,1,0),vrep=c(0,0,0,1))
contrasts(d$race)=cbind(vhisp=c(0,1,0,0),vblack=c(1,0,0,0),vother=c(0,0,1,0))
contrasts(d$polor)=cbind(vref=c(0,0,0,1),vlib=c(0,1,0,0),vcons=c(1,0,0,0))
contrasts(d$condmean) <- cbind(partially=c(0,0,1),fully=c(1,0,0))
contrasts(d$condstorm) <- c(1,0)

##### Numbers in 'Bounded Uncertainty with or without irreducible uncertainty' section and Table 2 #####

##### Message Acceptance

rs1 <- lm(seriousness ~ condstorm * condmean + coastdwell + 
          partyID + polor + gender + age + race + educ + income + 
          region, d, weights=weight1)
summary(rs1)
confint(rs1)

# Effect size for interactions - partially bounded * irreducible
rs1b<-lm(seriousness ~ high + both + condstorm + high:condstorm + both:condstorm + coastdwell + 
          partyID + polor + gender + age + race + educ + income + 
          region, d, weights=weight1)
rs1a<-lm(seriousness ~ high + both + condstorm + both:condstorm + coastdwell + 
           partyID + polor + gender + age + race + educ + income + 
           region, d, weights=weight1)
partial.R2(rs1a, rs1b)

# Effect size for interactions - fully bounded * irreducible
rs1b<-lm(seriousness ~ high + both + condstorm + high:condstorm + both:condstorm + coastdwell + 
           partyID + polor + gender + age + race + educ + income + 
           region, d, weights=weight1)
rs1a<-lm(seriousness ~ high + both + condstorm + high:condstorm + coastdwell + 
           partyID + polor + gender + age + race + educ + income + 
           region, d, weights=weight1)
partial.R2(rs1a, rs1b)

##### Trust in Scientists

rs2 <- glm(trustmed2 ~ condstorm * condmean + coastdwell + 
           partyID + polor + gender + age + race + educ + income + 
           region, d, weights=weight1, family=binomial(link=probit))
summary(rs2)
confint(rs2)
# Odd's ratios
exp(coef(rs2)) 
# McFadden's pseudo r-squared
PseudoR2(rs2, which = "McFadden")

##### Additional numbers in 'High partially bounded uncertainty' section #####

# Effect size - message acceptance
esc_B(-0.0086346, 0.3540315, 191, 196, es.type = c("d"))

##### Additional numbers in 'Fully bounded uncertainty without irreducible uncertainty' section #####

# Effect size - message acceptance
esc_B(0.0635027, 0.3540315, 191, 212, es.type = c("d"))

# Combining comparison group so that it is both no bounded uncertainty and partially bounded uncertainty

##### Message Acceptance

rs1 <- lm(seriousness ~ condstorm * condfully + coastdwell + 
            partyID + polor + gender + age + race + educ + income + 
            region, d, weights=weight1)
summary(rs1)
confint(rs1)

# Effect size - message acceptance
esc_B(0.066974, 0.3540315, 387, 212, es.type = c("d"))

##### Trust in Scientists

rs2 <- glm(trustmed2 ~ condstorm * condfully + coastdwell + 
             partyID + polor + gender + age + race + educ + income + 
             region, d, weights=weight1, family=binomial(link=probit))
summary(rs2)
confint(rs2)
# Odd's ratios
exp(coef(rs2)) 

##### Mediation

# Without irreducible uncertainty

contrasts(d$condstorm) <- c(0,1)

med.fit <- glm(trustmed2 ~ condstorm * condfully + coastdwell + 
                 partyID + polor + gender + age + race + educ + income + 
                 region, d, weights=weight1, family = binomial('probit'))
summary(med.fit)
exp(coef(med.fit)) # Odd's ratio

out.fit <- lm(seriousness ~ trustmed2 + condstorm * condfully + coastdwell + 
                partyID + polor + gender + age + race + educ + income + 
                region, weights=weight1, d)
summary(out.fit)

med.out1 <- mediate(med.fit, out.fit,  treat = "condfully", control.value = 'none/partial', treat.value = 'z fully bounded',
                   mediator = "trustmed2", sims=5000, covariates=list(condstorm='no irreducible'), boot=T)
summary(med.out1)

# With irreducible uncertainty

med.fit <- glm(trustmed2 ~ condstorm * condfully + coastdwell + 
                 partyID + polor + gender + age + race + educ + income + 
                 region, d, weights=weight1, family = binomial('probit'))
summary(med.fit)

out.fit <- lm(seriousness ~ trustmed2 + condstorm * condfully + coastdwell + 
                partyID + polor + gender + age + race + educ + income + 
                region, weights=weight1, d)
summary(out.fit)

med.out2 <- mediate(med.fit, out.fit,  treat = "condfully", control.value = 'none/partial', treat.value = 'z fully bounded',
                   mediator = "trustmed2", sims=5000, covariates=list(condstorm='irreducible'), boot=T)
summary(med.out2)

##### High partially bounded uncertainty with irreducible uncertainty #####

contrasts(d$condstorm) <- c(0,1)

##### Message Acceptance

rs1 <- lm(seriousness ~ condstorm * condmean  + coastdwell + 
            partyID + polor + gender + age + race + educ + income + 
            region, d, weights=weight1)
summary(rs1)
confint(rs1)

# Effect size - message acceptance
esc_B(-0.0151761, 0.3540315, 194, 176, es.type = c("d"))

##### Trust in Scientists

rs2 <- glm(trustmed2 ~ condstorm * condmean + coastdwell + 
             partyID + polor + gender + age + race + educ + income + 
             region, d, weights=weight1, family=binomial(link=probit))
summary(rs2)
confint(rs2)
# Odd's ratios
exp(coef(rs2)) 

##### Fully bounded uncertainty with irreducible uncertainty #####

contrasts(d$condstorm) <- c(0,1)

##### Message Acceptance

rs1 <- lm(seriousness ~ condstorm * condfully  + coastdwell + 
            partyID + polor + gender + age + race + educ + income + 
            region, d, weights=weight1)
summary(rs1)
confint(rs1)

# Effect size - message acceptance
esc_B(-0.101344, 0.3540315, 370, 198, es.type = c("d"))

##### Trust in Scientists

rs2 <- glm(trustmed2 ~ condstorm * condfully + coastdwell + 
             partyID + polor + gender + age + race + educ + income + 
             region, d, weights=weight1, family=binomial(link=probit))
summary(rs2)
confint(rs2)
# Odd's ratios
exp(coef(rs2)) 

##### Supplement #####

##### Supplemental Note 3 - Analyses omitting control variables #####

contrasts(d$condmean) <- cbind(partially=c(0,0,1),fully=c(1,0,0))

##### Message acceptance

# Without irreducible uncertainty

contrasts(d$condstorm) <- c(1,0)

r1 <- lm(seriousness ~ condmean * condstorm, d); anova(r1); summary(r1)
confint(r1)
etaSquared(r1)

r1 <- lm(seriousness ~ high + both + condstorm + high:condstorm + both:condstorm, d); anova(r1); summary(r1)
etaSquared(r1)

# With irreducible uncertainty

contrasts(d$condstorm) <- c(0,1)

r1 <- lm(seriousness ~ condmean * condstorm, d); anova(r1); summary(r1)
confint(r1)
etaSquared(r1)

r1 <- lm(seriousness ~ high + both + condstorm + high:condstorm + both:condstorm, d); anova(r1); summary(r1)
etaSquared(r1)

d$condfulltonone[d$condmean == 'fully bounded' & d$condstorm == 'no irreducible'] <- 'fully bounded'
d$condfulltonone[d$condmean == 'no bounds' & d$condstorm == 'no irreducible'] <- 'no uncertainty'
d$condfulltonone <- as.factor(d$condfulltonone)

d$condfulltononestorm[d$condmean == 'fully bounded' & d$condstorm == 'irreducible'] <- 'fully bounded'
d$condfulltononestorm[d$condmean == 'no bounds' & d$condstorm == 'irreducible'] <- 'no uncertainty'
d$condfulltononestorm <- as.factor(d$condfulltononestorm)

d$condhightonone[d$condmean == 'partially bounded' & d$condstorm == 'no irreducible'] <- 'high bounded'
d$condhightonone[d$condmean == 'no bounds' & d$condstorm == 'no irreducible'] <- 'no uncertainty'
d$condhightonone <- as.factor(d$condhightonone)

d$condhightononestorm[d$condmean == 'partially bounded' & d$condstorm == 'irreducible'] <- 'high bounded'
d$condhightononestorm[d$condmean == 'no bounds' & d$condstorm == 'irreducible'] <- 'no uncertainty'
d$condhightononestorm <- as.factor(d$condhightononestorm)

cohen.d(d$seriousness ~ d$condfulltononestorm)
cohen.d(d$seriousness ~ d$condfulltonone)
cohen.d(d$seriousness ~ d$condhightononestorm)
cohen.d(d$seriousness ~ d$condhightonone)

##### Trust in scientists

# Without irreducible uncertainty

contrasts(d$condstorm) <- c(1,0)

r1 <- glm(trustmed2 ~ condmean * condstorm, d, family=binomial(link=probit)); summary(r1)
confint(r1)
exp(coef(r1)) # Odd's ratios

# With irreducible uncertainty

contrasts(d$condstorm) <- c(0,1)

r1 <- glm(trustmed2 ~ condmean * condstorm, d, family=binomial(link=probit)); summary(r1)
confint(r1)
exp(coef(r1)) # Odd's ratios

# Mediation analysis

med.fit <- glm(trustmed2 ~ condmean * condstorm, data = d, family = binomial('probit')); summary(med.fit)
out.fit <- lm(seriousness ~ trustmed2 + condmean * condstorm, data = d); summary(out.fit)

med.out <- mediate(med.fit, out.fit,  treat = "condmean", control.value = 'no bounds', treat.value = 'fully bounded',
                   mediator = "trustmed2", covariates=list(condstorm='no irreducible'), sims=5000, boot=T)
summary(med.out)

med.fit <- glm(trustmed2 ~ condmean * condstorm, data = d, family = binomial('probit'))
out.fit <- lm(seriousness ~ trustmed2 + condmean * condstorm, data = d)

med.out <- mediate(med.fit, out.fit,  treat = "condmean", control.value = 'no bounds', treat.value = 'fully bounded',
                   mediator = "trustmed2", covariates=list(condstorm='irreducible'), sims=5000, boot=T)
summary(med.out)

##### Supplemental Table 3 #####

aggregate(seriousness ~ condmean + condstorm, d, mean)
aggregate(seriousness ~ condmean + condstorm, d, sd)

##### Supplemental note 5 - effect size of fully bounded vs. partially bounded uncertainty #####

# Message acceptance
contrasts(d$condmean) <- cbind(none=c(0,1,0),fully=c(1,0,0))
rs1 <- lm(seriousness ~ condstorm * condmean + coastdwell + 
            partyID + polor + gender + age + race + educ + income + 
            region, d, weights=weight1)
summary(rs1)
confint(rs1)
# Effect size - message acceptance
esc_B(0.0700443, 0.3540315, 199, 213, es.type = c("d"))

# Trust in scientists
rs2 <- glm(trustmed2 ~ condstorm * condmean + coastdwell + 
             partyID + polor + gender + age + race + educ + income + 
             region, d, weights=weight1, family=binomial(link=probit))
summary(rs2)
confint(rs2)
# Odd's ratio's
exp(coef(rs2)) 

##### Supplemental note 6 - effect size of fully bounded vs. partially bounded uncertainty, mediation #####

# Without irreducible uncertainty

contrasts(d$condstorm) <- c(1,0)
contrasts(d$condmean) <- cbind(none=c(0,1,0),fully=c(1,0,0))

med.fit <- glm(trustmed2 ~ condstorm * condmean + coastdwell + 
                 partyID + polor + gender + age + race + educ + income + 
                 region, d, weights=weight1, family = binomial('probit'))
summary(med.fit)

out.fit <- lm(seriousness ~ trustmed2 + condstorm * condmean + coastdwell + 
                partyID + polor + gender + age + race + educ + income + 
                region, weights=weight1, d)
summary(out.fit)

med.out <- mediate(med.fit, out.fit,  treat = "condmean", control.value = 'partially bounded', treat.value = 'fully bounded',
                   mediator = "trustmed2", sims=5000, covariates=list(condstorm='no irreducible'), boot=T)
summary(med.out)

##### Supplemental note 7 - sensitivity analyses #####

# Sensitivty analyses

medsens1 <- medsens(med.out1, effect.type='indirect', sims=1000)
summary(medsens1)

medsens2 <- medsens(med.out2, effect.type='indirect', sims=1000)
summary(medsens2)

##### Supplemental note 8 - moderation of the effects of uncertainty by party affiliation #####

d$Democrat <- ifelse(d$partyID == 'Democrat', 1, 0)
d$Republican <- ifelse(d$partyID == 'Republican', 1, 0)
d$Independent <- ifelse(d$partyID == 'Independent', 1, 0)
d$Refused <- ifelse(d$partyID == 'Refused', 1, 0)

contrasts(d$condstorm) <- c(1,0)

rs1<-lm(seriousness ~ high + both + condstorm + high:condstorm + both:condstorm + 
          Democrat + Independent + Refused + high:Democrat + high:Independent + high:Refused +
          both:Democrat + both:Independent + both:Refused +
          condstorm:Democrat + condstorm:Independent + condstorm:Refused +
          high:Democrat:condstorm + high:Independent:condstorm + high:Refused:condstorm +
          both:Democrat:condstorm + both:Independent:condstorm + both:Refused:condstorm +
          coastdwell + 
          polor + gender + age + race + educ + income + 
          region, d, weights=weight1)
summary(rs1)

rs1a<-lm(seriousness ~ high + both + condstorm + high:condstorm + both:condstorm + 
           Democrat + Independent + Refused + high:Democrat + high:Independent + high:Refused +
           both:Democrat + both:Independent + both:Refused +
           condstorm:Democrat + condstorm:Independent + condstorm:Refused +
           high:Democrat:condstorm + high:Independent:condstorm + high:Refused:condstorm +
           both:Independent:condstorm + both:Refused:condstorm +
           coastdwell + 
           polor + gender + age + race + educ + income + 
           region, d, weights=weight1)
summary(rs1a)

partial.R2(rs1a, rs1)

rs1<-lm(seriousness ~ condfully + condstorm + condfully:condstorm + 
          Democrat + Independent + Refused + 
          condfully:Democrat + condfully:Independent + condfully:Refused +
          condstorm:Democrat + condstorm:Independent + condstorm:Refused +
          condfully:Democrat:condstorm + condfully:Independent:condstorm + condfully:Refused:condstorm +
          coastdwell + 
          polor + gender + age + race + educ + income + 
          region, d, weights=weight1)
summary(rs1)

rs1a<-lm(seriousness ~ condfully + condstorm + condfully:condstorm + 
           Democrat + Independent + Refused + 
           condfully:Democrat + condfully:Independent + condfully:Refused +
           condstorm:Democrat + condstorm:Independent + condstorm:Refused +
           condfully:Independent:condstorm + condfully:Refused:condstorm +
           coastdwell + 
           polor + gender + age + race + educ + income + 
           region, d, weights=weight1)
summary(rs1a)

partial.R2(rs1a, rs1)

##### Supplemental note 9 - moderation of the effects of uncertainty by cognitive skills #####

d$educa[d$DM3=='High school graduate']<-'low'
d$educa[d$DM3=='Less than high school graduate']<-'low'
d$educa[d$DM3=='Technical/trade school']<-NA
d$educa[d$DM3=='Some college']<-NA
d$educa[d$DM3=='College graduate']<-'high'
d$educa[d$DM3=='Some graduate school']<-'high'
d$educa[d$DM3=='Graduate degree']<-'high'
d$educa <- as.factor(d$educa)

d$educb[d$DM3=='High school graduate']<-'moderate'
d$educb[d$DM3=='Less than high school graduate']<-'low'
d$educb[d$DM3=='Technical/trade school']<-'moderate'
d$educb[d$DM3=='Some college']<-'high'
d$educb[d$DM3=='College graduate']<-'high'
d$educb[d$DM3=='Some graduate school']<-'high'
d$educb[d$DM3=='Graduate degree']<-'high'
d$educb <- as.factor(d$educb)

d$educc[d$DM3=='High school graduate']<-'low'
d$educc[d$DM3=='Less than high school graduate']<-'low'
d$educc[d$DM3=='Technical/trade school']<-'moderate'
d$educc[d$DM3=='Some college']<-'moderate'
d$educc[d$DM3=='College graduate']<-'high'
d$educc[d$DM3=='Some graduate school']<-'high'
d$educc[d$DM3=='Graduate degree']<-'high'
d$educc <- as.factor(d$educc)

d$educd[d$DM3=='High school graduate']<-.17
d$educd[d$DM3=='Less than high school graduate']<-0
d$educd[d$DM3=='Technical/trade school']<-.33
d$educd[d$DM3=='Some college']<-.5
d$educd[d$DM3=='College graduate']<-.66
d$educd[d$DM3=='Some graduate school']<- .83
d$educd[d$DM3=='Graduate degree']<-1

# First coding - dichotomous

rs1<-lm(seriousness ~ condstorm + condfully + educa + 
          condstorm:condfully + condstorm:educa +
          condfully:educa +
          condfully:condstorm:educa +
          coastdwell + 
          partyID + polor + gender + age + race + income + 
          region, d, weights=weight1)
summary(rs1)

rs1a<-lm(seriousness ~ condstorm + condfully + educa + 
          condstorm:condfully + condstorm:educa +
          condfully:educa +
          coastdwell + 
          partyID + polor + gender + age + race + income + 
          region, d, weights=weight1)
summary(rs1a)

partial.R2(rs1a, rs1)

# Second coding - trichotomous 1

d$educbmoderate <- ifelse(d$educb == 'moderate', 1, 0)
d$educblow <- ifelse(d$educb == 'low', 1, 0)
d$educbhigh <- ifelse(d$educb == 'high', 1, 0)

rs1<-lm(seriousness ~ condstorm + condfully + educblow + educbmoderate + 
          condstorm:condfully + condstorm:educblow + condstorm:educbmoderate +
          condfully:educblow + condfully:educbmoderate +
          condfully:condstorm:educblow + condfully:condstorm:educbmoderate +
          coastdwell + 
          partyID + polor + gender + age + race + income + 
          region, d, weights=weight1)
summary(rs1)

rs1a<-lm(seriousness ~ condstorm + condfully + educblow + educbmoderate + 
           condstorm:condfully + condstorm:educblow + condstorm:educbmoderate +
           condfully:educblow + condfully:educbmoderate +
           condfully:condstorm:educblow +
           coastdwell + 
           partyID + polor + gender + age + race + income + 
           region, d, weights=weight1)
summary(rs1a)

partial.R2(rs1a, rs1)

rs1b<-lm(seriousness ~ condstorm + condfully + educblow + educbmoderate + 
           condstorm:condfully + condstorm:educblow + condstorm:educbmoderate +
           condfully:educblow + condfully:educbmoderate +
           condfully:condstorm:educbmoderate +
           coastdwell + 
           partyID + polor + gender + age + race + income + 
           region, d, weights=weight1)
summary(rs1b)

partial.R2(rs1b, rs1)

# Effect sizes
esc_B(0.0353469, 0.3540315, 480, 253, es.type = c("d"))
esc_B(0.076921, 0.3540315, 236, 131, es.type = c("d"))
esc_B(0.320488, 0.3540315, 40, 25, es.type = c("d"))

# Third coding - trichotomous 2

d$educcmoderate <- ifelse(d$educc == 'moderate', 1, 0)
d$educclow <- ifelse(d$educc == 'low', 1, 0)
d$educchigh <- ifelse(d$educc == 'high', 1, 0)

rs1<-lm(seriousness ~ condstorm + condfully + educclow + educcmoderate + 
          condstorm:condfully + condstorm:educclow + condstorm:educcmoderate +
          condfully:educclow + condfully:educcmoderate +
          condfully:condstorm:educclow + condfully:condstorm:educcmoderate +
          coastdwell + 
          partyID + polor + gender + age + race + income + 
          region, d, weights=weight1)
summary(rs1)

rs1a<-lm(seriousness ~ condstorm + condfully + educclow + educcmoderate + 
           condstorm:condfully + condstorm:educclow + condstorm:educcmoderate +
           condfully:educclow + condfully:educcmoderate +
           condfully:condstorm:educclow +
           coastdwell + 
           partyID + polor + gender + age + race + income + 
           region, d, weights=weight1)
summary(rs1a)

partial.R2(rs1a, rs1)

# Fourth coding - continuous

rs1<-lm(seriousness ~ condstorm + condfully + educd + 
          condstorm:condfully + condstorm:educd +
          condfully:educd + 
          condfully:condstorm:educd +
          coastdwell + 
          partyID + polor + gender + age + race + income + 
          region, d, weights=weight1)
summary(rs1)

rs1a<-lm(seriousness ~ condstorm + condfully + educd +
           condstorm:condfully+ condstorm:educd +
           condfully:educd +
           condfully:condstorm:educd +
           coastdwell + 
           partyID + polor + gender + age + race + income + 
           region, d, weights=weight1)
summary(rs1a)

partial.R2(rs1a, rs1)
