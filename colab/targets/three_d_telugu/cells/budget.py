import colab_budget
BUDGET = colab_budget.SessionBudget(MAX_SESSION_MINUTES)
# Include time already spent on preflight/install in the local session limit.
BUDGET.deadline = SESSION_STARTED + MAX_SESSION_MINUTES * 60
BUDGET.check('plan validation')
