import os

os.environ["CLEANLAB_API_BASE_URL"] = "https://api.dev-bc26qf4m.cleanlab.ai/api"
# os.environ["CLEANLAB_API_BASE_URL"] = "http://localhost:8500/api"

from cleanlab_studio import Studio
import pandas as pd


API_KEY = "350b3ee6fbe64d21a6012ea281ce0ca1"
MODEL_ID = "cea761848e5f449b85e34fe347696b53"
# API_KEY = "75f2ab8c962c40169917136756c5d937"
# MODEL_ID = "750dbdfb6549470192573b9646be40e9"
BATCH = pd.read_csv(
    "/Users/tony/test_files/text_amazon_reviews_test_small.csv", index_col=False, header=0
).loc[0, :]
print(BATCH)
# TEXT_BATCH = pd.Series([
#     "This magazine was great for the times but as with all other technology magazines the new stuff isn't as good a lot of advertisments and reviews seem biased.",
#     "We ordered this magazine for our grandson (then 7 going on 30) who was/is deploy into technology. He really enjoyed every issue.",
#     "I didn't receive a full year.  I only receive the magazine twice.  It's a good magazine, I just didn't receive it as promised.",
#     "I was hoping for more technical than what was there. it seems to be more like 'look how cool this is' than a technical publication. It's like sport compact car, but for computers.",
#     "I only received one copy of the mag so I couldn't really find out if it was good reading or not",
#     "This magazine is just ok. I ended up subscribing to pc world instead. They are more for the technician and not just the cusumer.",
#     "There articles are alright, but they screw you on the amount you get as i only got 10 of the 12 months subcription. so be carefull unless you are on the auto renew.",
#     "Excellent product! I love reading through the magazine and learning about the cool new products out there and the cool programs!",
#     "I ordered this hoping to learn more about the latest gadgets, and I did learn some things but in over my head over all.  I do not enjoy this reading at all.",
#     "Love the magazine.  The price through Amazon is well worth it for the knowledge recieved and the subscription process is painless",
#     "I bought this subscription for my son. He is presently building a computer. He said it has lots of good and useful information in it.",
# ], name="review_text")
studio = Studio(API_KEY)
model = studio.get_model(MODEL_ID)
results = model.predict(BATCH)
print(results)
