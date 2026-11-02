# How 3D works — Telugu script v4

**Draft for review.** Target: approximately 60–90 seconds; measure actual synthesized speech before rendering. This revision does not update the production notebook or implement new validators.

## sc01 — observation

**అర్జున్:** అమ్మో! ఆ popcorn నా మీద పడుతుందనుకున్నా!

**మీరా:** సినిమా చూస్తున్నావా, popcornకి దారి ఇస్తున్నావా?

**అర్జున్:** నవ్వకు! తెర మీద బొమ్మ నా దగ్గరికి వచ్చినట్టు ఎలా కనిపించింది?

## sc02 — question

**మీరా:** అద్దాలు అలాగే ఉంచు. ఒక్క కన్ను మూసి చూడు.

**అర్జున్:** నాకు దగ్గరికి వస్తున్నట్టు అనిపించడం తగ్గింది. కానీ బొమ్మలో ఇంకా కొంత లోతు కనిపిస్తోంది.

**మీరా:** అవును. వస్తువుల పరిమాణం లాంటి సూచనలతో ఒక్క కన్నుతోనూ లోతును అంచనా వేయగలం.

**అర్జున్:** అయితే రెండు కళ్లతో చూస్తే ఇంకేం మారుతుంది?

## sc03 — demonstration

**మీరా:** రెండు కళ్లూ కొంచెం వేర్వేరు స్థానాల్లో ఉంటాయి కదా. అందుకే ఒక్కో కంటికి కొంచెం వేరే బొమ్మ కనిపిస్తుంది.

**అర్జున్:** అంటే ఒక కన్ను మూస్తే, ఆ రెండో చూపు ఉండదన్నమాట?

**మీరా:** అవును. మెదడు రెండు కళ్ల బొమ్మల్లోని తేడాలను ఉపయోగించి లోతును గుర్తిస్తుంది.

## sc04 — correction

**అర్జున్:** మరి ఒకే తెర నుంచి ఒక్కో కంటికి వేరే బొమ్మ ఎలా వస్తుంది?

**మీరా:** పోలరైజ్డ్ త్రీడీ పద్ధతిలో, రెండు బొమ్మలు వేర్వేరు పోలరైజేషన్‌తో ఒకే తెరపై పడతాయి.

**అర్జున్:** అంటే ఈ అద్దాలు ఏ బొమ్మ ఏ కంటికి వెళ్లాలో వేరు చేస్తాయా?

**మీరా:** అవును. ఒక్కో లెన్స్‌లోని ఫిల్టర్ దానికి సరిపోయే బొమ్మను ఆ కంటికి అందిస్తుంది.

## sc05 — reveal

**అర్జున్:** ఓహో! బొమ్మ బయటికి రాలేదు. లోతు ఉన్నట్టు నాకు అనిపించింది!

**మీరా:** అర్థమైంది కదా! ఇప్పుడు సినిమా చూద్దామా?

## sc06 — closing

**అర్జున్:** ఈసారి మాత్రం తల వంచను.

**మీరా:** సరే, నీ ఛాలెంజ్ గుర్తుపెట్టుకుంటా!

**అర్జున్:** అనుకున్నాను గానీ… దగ్గరికి వచ్చినట్టే ఉంది!


## Production directions

- **Setting:** one modern cinema throughout, using the approved cinema/v1 environment. Keep the same seats, screen direction and glasses. No lobby teleport, spilled popcorn, object handoff or invisible prop action. Depth diagrams are explicitly illustrative.
- **Humor:** the opening original Telugu reaction (“సినిమా చూస్తున్నావా, popcornకి దారి ఇస్తున్నావా?”) and the closing duck are one setup/payoff, not a copied film dialogue or celebrity voice. No external meme clip or copyrighted soundtrack is needed.
- **Voice:** friendly conversational Telugu; Meera explains without lecturing and Arjun reacts without shouting. Separate spoken text is provided in JSON. Parler remains the backend; naturalness must be established by listening, not assumed from this rewrite.
- **Music:** curiosity genre, existing CPU procedural score, sparse opening pulse, lower intensity under the eye/filter explanation and a gentle resolving ending. The current score follows scene moods; precise word-aligned stingers and new adaptive scoring are not implemented here.
- **Checks implemented for this revision:** structural/story validation, known factual claim IDs, no unexplained location changes, no held props/handoffs, and no declared removal of glasses. Existing asset checks, measured-duration limit and output QC remain. General semantic, visual and pronunciation correctness still require review.
- **Output:** run the existing Telugu notebook filename; its new default project is `three_d_friends_te_v4_cinematic`. Old v3 outputs and approvals are not reused. This version needs the cinema background, not the lobby.
- **Release status:** draft. The updated notebook deliberately records the new Telugu script/voice review as UNKNOWN. There is no automatic claim that all facts, accent and acting have been human-approved.
