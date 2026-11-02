"""Local resource limits for the Telugu cinematic target, not Colab quota predictions."""
import time


class BudgetExceeded(RuntimeError):
    pass


class SessionBudget:
    def __init__(self, minutes=60, clock=time.monotonic):
        if not 0 < minutes <= 90:
            raise ValueError('Session budget must be between 0 and 90 minutes.')
        self.on_exhausted = None
        self.clock = clock
        self.deadline = clock() + minutes * 60

    def check(self, stage):
        if self.clock() >= self.deadline:
            if self.on_exhausted:
                self.on_exhausted()
            raise BudgetExceeded(f'Local session budget reached before {stage}. Completed files are on Drive; '
                                 'resume in a later allocated session with Run all.')

    def timeout(self, maximum=600):
        self.check('next operation')
        return max(0.01, min(maximum, self.deadline - self.clock()))


def validate_target(plan):
    if plan.get('language') != 'te' or plan.get('tts_backend') != 'indic_parler':
        raise ValueError('This target requires Telugu and local Indic Parler speech.')
    if not plan.get('strict_open_source'):
        raise ValueError('Hosted voices are disabled for this target.')
    shots = plan.get('shots', [])
    if not shots or len(shots) > 12:
        raise ValueError('Expected 1–12 drawn scenes for this short video.')
    if any(s.get('recipe') not in {'story', 'character', 'demo', 'diagram', 'card'} for s in shots):
        raise ValueError('This target permits drawn scenes only; AI video downloads are disabled.')
    if sum(len(s.get('lines', [])) for s in shots) > 60:
        raise ValueError('This short-video target permits at most 60 dialogue lines.')
    for s in shots:
        for v in s.get('story', {}).get('visuals', []):
            if v.get('recipe') not in {'character', 'demo', 'diagram', 'card'}:
                raise ValueError('Nested story visuals must also be drawn; no generated video.')
