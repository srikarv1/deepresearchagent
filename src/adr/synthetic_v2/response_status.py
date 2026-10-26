"""Provider termination diagnostics before parsing teacher JSON; never retries."""
def generation_failure(text, raw, allowance):
    choices=raw.get('choices') or []
    finish=choices[0].get('finish_reason') if choices else None
    usage=raw.get('usage') or {}
    completion=usage.get('completion_tokens')
    reasoning=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens')
    exhausted=isinstance(completion,int) and completion>=allowance
    reasoning_exhausted=isinstance(reasoning,int) and reasoning>=allowance
    empty=not text.strip()
    if not (empty or finish=='length' or exhausted or reasoning_exhausted):
        return None
    budget=finish=='length' or exhausted or reasoning_exhausted
    return {'structural_validation':'not_run','generation_status':'failed',
        'error_type':'GenerationBudgetFailure' if budget else 'EmptyAssistantContent',
        'finish_reason':finish,'empty_assistant_content':empty,
        'completion_allowance':allowance,'completion_tokens':completion,'reasoning_tokens':reasoning,
        'completion_exhausted':exhausted,'reasoning_exhausted':reasoning_exhausted,
        'raw_preserved':True,'automatic_regeneration':False}
