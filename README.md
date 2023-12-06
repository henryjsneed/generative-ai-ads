## The Future of Personalized Advertising: Integrating AI-Generated Ad Copy with Deep Learning Recommendation Systems

Ad copy may be the most important feature of static digital
advertisements but it is neither included in most ad click-through rate datasets nor integrated into recommendation systems like the Facebook Deep Learning Recommendation Model. DLRM and recommendation
systems broadly are not designed to interpret textual information but are instead adept at recommending content to
users based on latent patterns in copious amounts of user
behavior and interaction data. This limitation makes it impossible to recommend increasingly effective ads to users
on the basis of ad copy and thus impedes the development
of a system that iteratively generates new ad copy that appeals to users unique, individual preferences. I propose a
solution that adapts DLRM to interpret textual information
and integrates it with a Large Language Model to deliver personalized ad copy to users. This method transcends
DLRM’s limitations, enabling more direct and resonant ad
creation. Additionally, the proposed architecture addresses
the issue of repetitive ads since new ad copy can be generated for each new ad impression. By successfully modifying
Facebook DLRM to recommend content informed by textual
data, this work validates the viability of the proposed ad
technology application.

The code for the training and testing of the model are located in **src/train_dlrm.ipynb** and **src/test_dlrm.ipynb**, respectively.
