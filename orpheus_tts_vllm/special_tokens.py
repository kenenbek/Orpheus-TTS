
tokeniser_length = 128256
SOT = 128000 # start_of_text
EOT = 128009 # end_of_text

SOS = tokeniser_length + 1 # start_of_speech
EOS = tokeniser_length + 2 # end_of_speech

SOH = tokeniser_length + 3 # start_of_human
EOH = tokeniser_length + 4 # end_of_human

SOA = tokeniser_length + 5 # start_of_ai
EOA =  tokeniser_length + 6 # end_of_ai
PAD_TOKEN = tokeniser_length + 7 # pad_token

ATS = tokeniser_length + 10 # audio_tokens_start