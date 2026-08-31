#Data clean for main Bovitz data
#Reed Orchinik

#Clear
rm(list = ls())

#Load packages
library('dplyr')
library('stringr')
library('reshape2')
library('tidyr')
library('ggplot2')
library('tidyverse')

#Load data and exclude test responses and preview responses
df_full <- read.csv2('../data/final_bovitz_raw.csv', sep = ",")

df_full <- df_full %>%
  filter(ResponseId != "THROWOUT" & ResponseId != "THROW OUT" &
           Status == 0)

#Mark people who fail attention checks
#Fails is a dummy for if they fail either the simple attention check or captcha
#Cons_attentive is a dummy for if they pass all 5 consensus attention checks
df_full <- df_full %>%
  mutate(fails = as.numeric((df_full$nick != 5) | df_full$captcha != "15"),
         cons_attentive = (between(cons.attention.check_1, 20, 30) &
                             between(cons.attention.check_4, 20, 30) &
                             between(cons.attention.check_5, 20, 30) &
                             between(cons.attention.check_6, 20, 30) &
                             between(cons.attention.check_7, 20, 30)))

#Rename variables
df_full <- df_full %>%
  rename(prior_consensus_num = prior.consensus_1,
         prior_consensus_num_conf = consensus.conf_1,
         prior_cc_occur = prior.CC.occurring_1,
         prior_cc_occur_conf = prior.CC.conf_1,
         prior_sci_biased = Prior.Sci.Biased_1,
         prior_sci_biased_yes = Sci.Bias.Direction_1,
         P_E_yes_given_cc_unbiased = Ev.Yes.Given.CC_1,
         P_E_no_given_no_cc_unbiased = Ev.No.Given.No.CC_1,
         P_cc_given_cons50 = cc.belief.given.cons_1,
         P_cc_given_cons75 = cc.belief.given.cons_4,
         P_cc_given_cons90 = cc.belief.given.cons_5,
         P_cc_given_cons97 = cc.belief.given.cons_6,
         P_cc_given_cons99 = cc.belief.given.cons_7,
         P_pro_bias_given_cons50 = pro.trust.given.cons_1,
         P_pro_bias_given_cons75 = pro.trust.given.cons_4,
         P_pro_bias_given_cons90 = pro.trust.given.cons_5,
         P_pro_bias_given_cons97 = pro.trust.given.cons_6,
         P_pro_bias_given_cons99 = pro.trust.given.cons_7,
         P_anti_bias_given_cons50 = ant.trust.given.cons_1,
         P_anti_bias_given_cons75 = ant.trust.given.cons_4,
         P_anti_bias_given_cons90 = ant.trust.given.cons_5,
         P_anti_bias_given_cons97 = ant.trust.given.cons_6,
         P_anti_bias_given_cons99 = ant.trust.given.cons_7,
         P_pro_skill_given_cons50 = pro.skill.given.cons_1,
         P_pro_skill_given_cons75 = pro.skill.given.cons_4,
         P_pro_skill_given_cons90 = pro.skill.given.cons_5,
         P_pro_skill_given_cons97 = pro.skill.given.cons_6,
         P_pro_skill_given_cons99 = pro.skill.given.cons_7,
         P_anti_skill_given_cons50 = ant.skill.given.cons_1,
         P_anti_skill_given_cons75 = ant.skill.given.cons_4,
         P_anti_skill_given_cons90 = ant.skill.given.cons_5,
         P_anti_skill_given_cons97 = ant.skill.given.cons_6,
         P_anti_skill_given_cons99 = ant.skill.given.cons_7)

#Clean belief shift variables
df_full <- df_full %>%
  mutate(belief_shift_climate = case_when(condition == "skill" ~ as.factor(Skill.shift_1),
                                          condition == "trust" ~ as.factor(Trust.shift_1)),
         belief_shift_unbiased = case_when(condition == "skill" ~ as.factor(Skill.shift_2),
                                           condition == "trust" ~ as.factor(Trust.shift_2)),
         belief_shift_skill = case_when(condition == "skill" ~ as.factor(Skill.shift_3),
                                        condition == "trust" ~ as.factor(Trust.shift_3)))

#Drop people with missing values for any of the consensus info
varlist <- c("cc_given_cons", "pro_bias_given_cons", "anti_bias_given_cons", "pro_skill_given_cons", "anti_skill_given_cons")
numlist <- c("50", "75", "90", "97", "99")
df_full$flag = 0
for(var in varlist){
  for(num in numlist){
    df_full$flag <- df_full$flag + is.na(df_full[[sprintf("P_%s%s", var, num)]])
  }
}

#Create variables for the 3 directions of scientist bias
df_full <- df_full %>%
  mutate(prior_sci_always_E_yes = prior_sci_biased*prior_sci_biased_yes/100,
         prior_sci_always_E_no = prior_sci_biased*(1-prior_sci_biased_yes/100),
         prior_sci_unbiased = 100 - (prior_sci_always_E_yes + prior_sci_always_E_no))

#Create drop variable
df_full$drop = (df_full$fails > 0 | df_full$flag > 0)

#Create new df with just relevant data
df <- subset(df_full, select = c("StartDate", "Duration..in.seconds.", "age", "gender", "race",
                                 "edu", "income", "god", "party", "politics", "politics_social", "politics_econ",
                                 "affpol_thermom_1", "affpol_thermom_2", "gov.trust", "pol.party.trust",
                                 "uni.science.trust", "priv.science.trust", "prior_consensus_num", "prior_consensus_num_conf", "prior_cc_occur", "prior_cc_occur_conf",
                                 "prior_sci_always_E_yes", "prior_sci_always_E_no", "prior_sci_unbiased", "P_E_yes_given_cc_unbiased", "P_E_no_given_no_cc_unbiased",
                                 "P_cc_given_cons50", "P_cc_given_cons75", "P_cc_given_cons90", "P_cc_given_cons97", "P_cc_given_cons99",
                                 "P_pro_bias_given_cons50", "P_pro_bias_given_cons75", "P_pro_bias_given_cons90", "P_pro_bias_given_cons97", "P_pro_bias_given_cons99",
                                 "P_anti_bias_given_cons50", "P_anti_bias_given_cons75", "P_anti_bias_given_cons90", "P_anti_bias_given_cons97", "P_anti_bias_given_cons99",
                                 "P_pro_skill_given_cons50", "P_pro_skill_given_cons75", "P_pro_skill_given_cons90", "P_pro_skill_given_cons97", "P_pro_skill_given_cons99",
                                 "P_anti_skill_given_cons50", "P_anti_skill_given_cons75", "P_anti_skill_given_cons90", "P_anti_skill_given_cons97", "P_anti_skill_given_cons99",
                                 "condition", "belief_shift_climate", "belief_shift_skill", "belief_shift_unbiased", "fails", "flag", "drop", "cons_attentive"))

#Create unique ID
df <- df %>%
mutate(ID = row_number())

#Clean trust variables
df <- df %>%
  dplyr::mutate(uni_sci_trust = case_when(uni.science.trust == "None at all" ~ 0,
                                          uni.science.trust == "Not very much confidence" ~ 1,
                                          uni.science.trust == "Quite a lot of confidence" ~ 2,
                                          uni.science.trust == "A great deal of confidence" ~ 3),
                priv_sci_trust = case_when(priv.science.trust == "None at all" ~ 0,
                                           priv.science.trust == "Not very much confidence" ~ 1,
                                           priv.science.trust == "Quite a lot of confidence" ~ 2,
                                           priv.science.trust == "A great deal of confidence" ~ 3))

#Create adjusted versions of main variables that change 0 and 100
df <- df %>%
  mutate(prior_cc_occur_adj = case_when(prior_cc_occur == 0 ~ 0.497462,
                                        prior_cc_occur == 100 ~ 99.50254,
                                        TRUE ~ as.numeric(as.character(prior_cc_occur))),
         prior_consensus_num_adj = case_when(prior_consensus_num == 0 ~ 0.497462,
                                             prior_consensus_num == 100 ~ 99.50254,
                                             TRUE ~ as.numeric(as.character(prior_consensus_num))),
         prior_sci_unbiased_adj = case_when(prior_sci_unbiased == 0 ~ 0.497462,
                                            prior_sci_unbiased == 100 ~ 99.50254,
                                            TRUE ~ as.numeric(as.character(prior_sci_unbiased))),
         prior_sci_skill_adj = case_when(P_E_yes_given_cc_unbiased == 0 ~ 0.497462,
                                         P_E_yes_given_cc_unbiased == 100 ~ 99.50254,
                                         TRUE ~ as.numeric(as.character(P_E_yes_given_cc_unbiased))))

#Create better looking party variable
df <- df %>%
  mutate(Party = case_when(party == 1 ~ "Dem",
                           party > 2 ~ "Ind",
                           party == 2 ~ "Rep"))


write.csv(df,"../data/final_clean.csv", row.names = FALSE)

