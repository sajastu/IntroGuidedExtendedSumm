

def get_sect_kws(dataset, section):
    KWs_str = ''
    if dataset=='longsumm':
        if section=='abstract':
            KWs_str = "[abstract]"
        if section == 'introduction':
            KWs_str = "[introduction, introduction and motivation, motivation, motivations, basics and motivations]"
        elif section=='conclusion':
            KWs_str = "[conclusion, conclusions, conclusion and future work, conclusions and future work, conclusion & future work, extensions, future work, related work and discussion, discussion and related work, conclusion and future directions, summary and future work, limitations and future work, future directions, conclusion and outlook, conclusions and future directions, conclusions and discussion, discussion and future directions, conclusions and discussions, conclusion and future direction, conclusions and future research, conclusion and future works, future plans, summary]"

        INTRO_KWs_STR = "[" + str(['' + kw.strip() + '' for kw in KWs_str[1:-1].split(',')]) + "]"
        return eval(INTRO_KWs_STR)[0]
