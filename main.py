import test
import hyperparameters as hp
import pygame
import cv2
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean

window = pygame.display.set_mode((hp.WIDTH, hp.HEIGHT))
pygame.display.set_caption('Go Reader')

# 0: empty
# 1: black
# 2: white
expected = []

board_info = [
    "AW[ae][af][ag][ai][al][an][bd][be][bg][bi][bj][bm][bo][cj][cn][co][di][dl][dm][dn][ed][eh][ei][el][en][eo][ep][fi][fj][fk][fq][gf][gh][gi][gk][gn][go][gq][gr][gs][hc][hg][hh][hi][hj][ho][hq][ic][id][ie][ii][il][im][ip][iq][jb][jc][jf][jm][jn][jo][jp][kc][kf][kg][kh][ki][kj][km][ko][lc][lg][li][lk][ll][lm][lr][ls][mc][md][mg][mi][mk][mn][ms][na][nb][ne][nf][ni][nj][nl][nm][nn][no][np][nq][nr][ns][oc][of][oh][ok][op][pb][pc][pd][pf][ph][pi][pj][pk][pp][qc][qe][qf][qh][qj][ql][qp][qq][qr][rd][rj][rq][sd]AB[ah][ao][ap][bc][bf][bh][bp][cd][ce][cg][ch][ci][cp][dc][dd][dh][do][dp][dq][dr][ee][ef][eg][em][eq][es][fb][fc][fe][fg][fh][fl][fm][fn][fo][fr][fs][ga][gc][ge][gg][gl][ha][hb][hd][he][hf][hk][hl][hm][hn][hp][ia][ib][if][ig][ih][ij][ik][in][io][ir][is][ja][jg][jh][ji][jj][jl][jq][jr][kb][kk][kn][kp][kr][ks][la][lb][lh][ln][lo][lq][mb][mh][mo][mp][mq][mr][ng][nh][ob][og][ol][oo][pg][pl][pm][po][qg][qi][qk][qn][qo][re][rf][rg][rh][ri][rk][rl][rn][rp][rr][se][sg][sh][sj][sl][sp][sq][ss]",
    "AW[ac][af][aj][ak][bb][bd][be][bf][bg][bi][bj][bl][bm][br][bs][ca][cc][ce][cf][ck][cl][cm][co][cq][cr][da][db][dc][de][df][dn][do][dp][ea][ee][ef][eg][eh][ei][ej][eo][ep][eq][fe][ff][fl][fo][fp][gd][ge][gf][gk][gl][gm][gn][go][hc][hf][hg][hk][hn][hp][ic][ie][il][im][in][jc][je][jf][jm][jr][js][kb][kd][kg][ki][km][kn][ld][lg][lh][li][lj][lk][lm][ln][lr][ls][mc][md][me][mf][mj][mq][ms][nd][ng][nh][ni][nj][nk][nn][nq][nr][oc][od][oe][of][oj][oq][or][os][pd][pf][pg][pn][pq][pr][ps][qe][qf][qg][qh][qi][qj][ql][qn][qp][qq][qr][rf][rh][rk][rl][rm][rn][rp][sh][sm]AB[ag][ah][ai][am][ba][bh][bn][cd][cg][ch][ci][cj][cn][cs][dd][dg][dh][di][dj][dk][dl][dm][dq][dr][ds][eb][ec][ed][ek][el][em][en][er][es][fa][fb][fd][fg][fh][fi][fj][fk][fm][fn][fq][fs][ga][gc][gg][gj][gp][gq][gr][gs][hb][hd][he][hh][hi][hj][ho][ib][if][ig][ih][ij][ik][io][jb][jg][jh][ji][jk][jl][jn][jo][jq][kc][kf][kh][kj][kk][kl][ko][kp][kq][kr][ks][lb][lc][lf][ll][lo][lp][lq][mb][mi][mk][ml][mm][mn][mp][nb][nc][nf][nl][nm][np][ob][og][oh][oi][ok][on][oo][op][pc][ph][pi][pj][pk][pl][pm][po][pp][qb][qc][qd][qk][qm][qo][qs][rc][re][rg][ro][rq][rr][rs][sd][se][sf][sg][sn][so][sp]",
    "AW[cd][ce][dc][dp][ed][fq][jd][kd][ld][md][nc][ob][pb][pc][qb]AB[cf][cn][de][df][dj][ke][le][me][nd][oc][od][pd][pl][po][qc][qf]",
    "AB[cd][dj][dn][dp][ec][eq][fp][go][gp][hn][in][jd][jm][jo][kg][ko][kp][mn][ng][nh][np][nq][ob][oc][oe][og][op][pd][pe][pf][pl][po][qi][qj][qk][ql][qm][qo][rl][rn][rp]AW[fo][fq][gm][gn][gq][ho][hq][io][jp][jq][kq][ln][lp][mm][mo][nc][nd][ne][ni][od][oh][oi][on][oo][oq][or][pc][pg][pk][pm][pp][qc][qd][qe][qg][qp][qr][rk][rm][rq][sl]",
    "AW[bn][bo][cg][cm][co][cq][dj][dp][ep][fq][oq][pf][pi][pq][qf][qn][qo][qp]AB[bp][bq][cd][cn][cp][do][dq][dr][ed][ic][jp][mp][nd][nq][oj][op][pd][po][qe]",
    "AB[bb][bd][bp][cc][ce][cf][cg][cn][co][db][de][dg][dj][do][eg][ek][en][ep][fb][fc][fd][ff][fp][fr][gf][gi][gj][gk][gl][gm][gn][gq][gs][hc][hd][he][hf][hh][hk][hn][hp][hr][ib][id][ij][iq][jb][jh][ji][jj][jp][jr][kb][kc][ki][kk][kn][ld][le][li][lk][lo][md][mf][mi][mk][ml][mm][mo][nc][nf][nj][no][np][ob][oc][of][oh][oi][oj][os][pb][pg][pk][pl][po][pp][pr][ps][qc][qd][qe][qg][qh][qj][qq][rf][rh][ri][rj][ro][rp][sg]AW[be][bm][bq][br][cd][ci][cj][cp][cr][dd][df][di][dl][dn][dp][dr][eb][ec][ee][ef][eq][er][fa][fe][fg][fh][fi][fj][fl][fq][gb][gc][gd][ge][gh][gp][hb][hi][hj][hm][ic][ie][ig][ih][ii][ik][im][io][jc][jd][je][jg][jk][jm][jo][kf][kh][kj][ko][kp][kq][kr][lb][lf][ll][lm][ln][mc][me][mg][mp][mr][nb][nd][ng][nh][nk][nq][ns][oe][og][ok][om][oq][or][pc][pd][pe][pf][pm][pn][qf][qk][ql][qn][qo][rk][rm][rn]",
    "AB[bb][bd][bp][cc][ce][cf][cg][cn][co][db][de][dg][dj][do][eg][ek][en][ep][fb][fc][fd][ff][fp][fr][gf][gi][gj][gk][gl][gm][gn][gq][gs][hc][hd][he][hf][hh][hk][hn][hp][hr][ib][id][ij][iq][jb][jh][ji][jj][jp][jr][kb][kc][ki][kk][kn][ld][le][li][lk][lo][md][mf][mi][mk][ml][mm][mo][nc][nf][nj][no][np][ob][oc][of][oh][oi][oj][os][pb][pg][pk][pl][po][pp][pr][ps][qc][qd][qe][qg][qh][qj][qq][rf][rh][ri][rj][ro][rp][sg]AW[be][bm][bq][br][cd][ci][cj][cp][cr][dd][df][di][dl][dn][dp][dr][eb][ec][ee][ef][eq][er][fa][fe][fg][fh][fi][fj][fl][fq][gb][gc][gd][ge][gh][gp][hb][hi][hj][hm][ic][ie][ig][ih][ii][ik][im][io][jc][jd][je][jg][jk][jm][jo][kf][kh][kj][ko][kp][kq][kr][lb][lf][ll][lm][ln][mc][me][mg][mp][mr][nb][nd][ng][nh][nk][nq][ns][oe][og][ok][om][oq][or][pc][pd][pe][pf][pm][pn][qf][qk][ql][qn][qo][rk][rm][rn]",
    "AB[ad][ai][aj][am][bb][bd][be][bj][bk][bl][bn][cc][cj][cl][cm][co][cp][cq][da][db][dc][dd][de][dh][di][dj][dm][dr][ea][ed][ee][ej][ek][el][eq][er][fd][fe][fg][fh][fj][fq][ge][gg][gh][gi][gk][gp][gr][hc][hd][he][hi][hk][hl][ho][hp][hq][hs][ia][ih][ii][ik][io][ip][iq][ir][is][ja][jb][jj][jk][jl][jr][kc][kf][kh][kj][kp][kr][lb][ld][le][lf][lh][li][ln][lo][lp][lq][ma][mc][mf][mg][mh][mi][mj][mk][mn][na][nb][nd][nf][nn][nq][oa][od][of][oh][oi][om][os][pe][pf][pg][ph][pi][pj][pm][po][pq][pr][ps][qe][qf][qh][qi][qm][qn][qo][qq][qs][rf][rl][rm][rp][rr][sm][so]AW[ae][af][ah][bf][bg][bi][cd][ce][cf][ch][ci][cn][df][dg][dn][do][dp][dq][eb][ec][ef][eg][eh][ei][em][eo][ep][fa][fc][ff][fi][fk][fl][fm][fo][fp][fr][gb][gc][gd][gf][gl][gm][go][ha][hb][hf][hg][hh][hm][hn][ib][ic][id][ie][ig][ij][il][im][in][jc][je][jf][jg][jh][ji][jm][jo][jp][jq][js][kd][ke][kg][ki][kk][kl][km][kn][ko][kq][ks][lg][lj][lk][lm][lr][ls][md][ml][mo][mp][mq][mr][ms][nc][ng][nh][ni][nj][nk][nl][nm][no][ns][ob][oc][og][oj][ol][on][oo][op][oq][or][pa][pb][pd][pk][pl][pn][pp][qb][qd][qg][qj][qk][ql][qp][rb][rc][re][rg][rh][ri][rk][ro][sd][sf][sk][sl]",
    "AB[ab][ac][ad][ba][be][bf][bh][bn][br][bs][cc][cd][cg][co][da][db][dc][dh][di][dn][do][dp][dq][eb][ei][ek][em][eo][eq][er][fb][ff][fh][fi][fm][fs][ga][gb][gi][gj][gk][gm][gn][gq][gr][hb][hj][hl][hm][hr][ii][ij][ik][im][iq][ji][jj][jk][jm][jn][jq][kd][ke][kq][kr][lf][ll][lm][lq][md][mf][mh][mk][ml][mn][mo][mr][ne][nl][nm][nn][no][nr][ob][oc][od][ok][ol][om][oo][op][oq][or][pe][pg][pl][pp][qe][qf][qj][qo][rf][rn][ro][rp][sn]AW[bo][bp][ce][cf][cq][cr][dd][df][dg][ec][ed][ee][ef][eg][eh][en][ep][fc][fg][fj][fn][fo][fp][fq][fr][gc][gf][gh][go][gp][ha][hc][hf][hg][hh][hi][hn][hp][hq][ia][ib][if][ih][il][in][ip][jb][jf][jh][jl][jo][jp][kf][ki][kj][kk][kl][km][kn][kp][lc][ld][le][lg][lh][lj][lk][ln][lo][lp][me][mi][mj][mp][mq][nf][ng][nh][nj][nk][np][nq][oe][of][oj][on][pb][pc][pd][pf][pj][pk][pm][pn][po][qd][qg][qh][qk][ql][qm][qn][qp][rd][re][rm][sm]",
    "AW[bo][bp][bq][br][cd][cf][cm][cn][co][cp][cr][cs][db][dc][de][dj][dl][dm][do][ds][eb][ef][eg][eh][ej][ek][fb][fh][fi][gb][gi][hc][hd][hf][hg][hh][hi][hj][hl][ia][ib][ic][if][ii][ij][ik][il][ja][jc][jd][jf][jj][jk][jl][kd][kf][kg][ki][kj][kl][km][ld][li][ll][lm][md][me][ml][mn][nj][nk][nl][nn][nq][oe][ok][om][on][os][pe][pf][pi][pn][po][pr][ps][qf][qg][qh][qi][qj][qm][qn][qp][qq][qr][rd][re][rj][rl][rn][ro][sc][sd][sf][si][sl][so]AB[cq][dd][dn][dp][dq][dr][ec][ed][ee][el][em][en][eo][ep][er][es][fc][fd][fe][ff][fg][fj][fk][fl][fo][fp][gc][gd][ge][gf][gg][gh][gj][gk][gl][he][hk][hm][id][ie][ig][ih][im][jb][je][jg][jh][ji][jm][jp][ka][kb][kc][ke][kh][kk][kn][lc][le][lf][lg][lh][lj][lk][ln][mc][mf][mh][mi][mj][mk][mo][nd][ne][ng][ni][nm][no][nr][ns][od][of][og][oi][oj][ol][oo][op][oq][or][pd][pg][ph][pj][pk][pl][pm][pp][pq][qc][qd][qe][qk][ql][ra][rc][rk][sb][sj][sk]",
    "AW[ac][aj][bb][bd][bi][bm][bp][br][bs][ca][cd][ch][ci][cj][ck][cq][cs][dd][dh][di][do][dp][ec][ed][ei][en][ep][fd][fe][fh][fj][fn][ge][gh][gk][gl][gm][gp][gq][gr][he][hf][hh][hi][hk][hm][hp][ig][ih][ij][im][in][io][ir][jb][jc][je][jh][jj][jk][ka][kb][kd][ke][ki][kj][kl][lc][le][lg][lh][lk][ll][lm][ls][me][mf][mg][mh][mj][mk][ml][ms][nc][ne][ng][nk][nm][nr][ns][oe][ok][om][oq][os][pc][pd][pk][pl][pq][pr][qd][qe][qk][qm][qq][rd][rm][rn][rp][rr][sd][sl][sm][sn][so][sp][sq]AB[ah][ai][bc][be][bg][bh][cb][cc][cf][cg][cr][db][dc][dg][dj][dl][dq][dr][ds][eb][eg][eh][ej][eo][eq][fc][fg][fi][fl][fo][fp][fq][fr][gb][gd][gg][gi][gn][go][hd][hg][hl][hn][ho][hq][hr][ia][ib][ic][id][ie][if][ik][il][ip][iq][ja][jd][jf][jg][jl][jm][jn][jo][jp][kf][kg][kh][km][kr][ks][lf][li][lj][ln][lr][mi][mm][mn][mq][mr][nf][nh][ni][nj][nn][nq][of][og][oj][on][op][pe][pf][pj][pm][pn][pp][qc][qf][qj][ql][qn][qo][qp][rc][re][rf][rj][rk][rl][ro][se][sk]",
    "AW[ai][aq][bi][bj][br][cd][cf][ch][cj][co][cs][da][db][dc][dd][di][dn][do][dp][dq][dr][ec][ee][ef][eg][eh][ei][ej][el][em][en][fg][fl][gf][gg][gh][gk][hg][hh][hj][hk][hl][hm][hn][hp][hq][ii][il][io][ip][iq][ir][jg][ji][jj][jk][jn][jq][js][kc][kd][ke][kf][kg][kh][kj][kl][km][kq][la][ld][lg][lh][li][lm][ma][mb][mc][me][mf][mh][mk][ml][mm][mr][na][nb][nk][nn][no][oa][ok][op][pa][pb][pl][pm][po][pp][pr][qa][qc][qd][qe][qm][qq][rb][rd][rf][rk][rm][rr][se][sf][sg][sh][sq][sr]AB[aj][ak][bk][bm][bo][bp][bq][cb][cc][ck][cn][cp][cq][cr][dj][dk][dl][dm][ea][eb][ed][ek][er][fb][fc][fd][fe][ff][fh][fi][fj][fk][fq][ge][gi][gj][gl][gm][gn][go][gp][gq][gr][hf][hi][ho][hr][hs][if][ig][ih][im][in][is][jb][jc][jd][je][jf][jh][jo][jp][ka][kb][kk][lb][lc][lj][lk][ll][ln][lp][md][mg][mi][mj][nc][nd][ne][nf][ng][nh][nj][nl][nm][ob][oc][oj][ol][om][on][oo][pc][pd][pe][pj][pk][pn][qf][qg][qk][ql][qn][qo][qp][rg][rh][ri][rl][rn][rp][rq][si][sp]",
    "AW[an][ao][be][bh][bi][bj][bn][cb][cc][cd][ce][cf][cg][cj][cl][cn][co][da][dd][de][dg][dk][dm][dn][ea][eb][ec][ek][el][en][fc][ff][fg][fh][fi][fj][fk][fn][fo][gc][gg][gk][gl][hf][hg][hi][hm][ie][if][ih][ii][ij][ja][jd][ji][jj][jk][jm][jn][jo][js][ka][kb][kc][kd][ke][kj][kk][ko][kp][ks][la][lc][le][lj][lk][lm][lo][lp][ma][me][mi][mk][ml][mm][mn][mo][mp][ms][na][nd][ne][nf][ng][nh][ni][nk][no][np][nr][ns][oa][oc][od][oi][ol][om][oo][oq][os][pd][pe][pj][pk][pl][po][pq][pr][qd][qe][ql][qq][qs][re][rp][rr][sd][se][sf][so][sp][sq]AB[ap][bf][bg][bo][bp][bq][ch][ci][cp][df][dh][dj][do][dq][ed][ee][ef][eg][eh][ei][ej][em][eo][ep][fa][fb][fd][fe][fl][fm][fp][gb][gd][ge][gf][gh][gi][gj][gm][gn][go][hb][hc][hd][he][hh][hj][hk][hl][ia][ib][id][ig][ik][il][im][in][io][ip][jb][jc][je][jf][jg][jh][jl][jp][jq][ki][kl][km][kn][kq][kr][lb][ld][ll][ln][lq][lr][ls][mb][mc][md][mq][mr][nb][nc][nl][nm][nn][nq][ob][oe][of][og][oh][on][op][pa][pb][pc][pf][ph][pi][pm][pn][pp][qb][qc][qf][qg][qj][qk][qm][qo][qp][ra][rc][rd][rf][rh][rj][rl][rm][rn][ro][sc][sg][si][sk][sn]",
    "AB[ab][ad][ak][an][bc][bd][be][bf][bg][bh][bj][bk][bl][bm][bn][ca][cb][cc][cd][cf][cg][cj][cl][db][dg][dh][dk][dl][dm][dn][do][ee][ej][el][em][en][eo][fb][fc][fd][fe][fj][fm][fo][fp][fr][ga][gc][ge][gj][gk][go][gr][hb][hd][he][hj][hl][hm][ho][hp][hq][ia][id][ii][im][io][iq][jb][jc][jl][jm][jo][jp][ka][kc][kk][kl][lb][lf][lh][li][lj][lk][ll][ln][mb][mc][md][mf][mi][mk][mm][mn][mo][na][nb][nd][ne][nf][nh][ni][nk][nm][ob][oc][ok][om][oo][op][or][pc][pd][pe][pj][pk][pn][po][pq][pr][ps][qf][qg][qh][qi][qj][qp][qr][rg][rh][ri][sh]AW[ae][af][ag][ah][ai][aj][ao][ap][bi][bo][bq][ce][ch][ci][cm][cn][co][cq][da][dc][dd][de][df][di][dj][dp][dq][ea][eb][ec][ed][ef][eg][eh][ei][ek][ep][eq][er][fa][ff][fk][fl][fn][fq][gf][gl][gm][gn][gp][gq][hf][hk][hn][hr][ie][if][ig][ij][ik][il][in][ir][jd][jh][ji][jj][jk][jn][jq][jr][kd][kf][kg][kh][ki][kj][km][kn][ko][kp][lc][ld][le][lg][lm][lo][lq][me][mg][mh][mj][ml][mp][ng][nj][nn][no][np][nq][nr][ns][oa][od][oe][of][og][oh][oi][oj][on][oq][os][pa][pb][pf][pg][ph][pi][qa][qc][qd][qe][rb][rd][rf][sc][se][sf][sg][si][sj]",
    "AB[ac][ah][ai][aj][bc][bd][bi][bk][cc][ci][ck][dc][dj][dk][dl][dm][eb][ec][ed][eh][el][en][ep][fa][fb][fh][fk][fm][fo][fp][fq][fs][ga][gf][gg][gh][gi][gj][gk][go][gq][gr][gs][hh][ho][hr][ih][ij][ip][jb][ji][jo][jp][jq][ka][kb][ki][ko][kp][lb][le][lh][ln][mc][me][mf][mg][mh][mn][mo][mp][na][nb][nd][ne][no][np][ob][oc][of][og][oq][pc][pd][pe][pf][pn][pp][pq][qg][qh][qp][re][rf][rg][rm][rn][ro][sf][sm]AW[ad][ae][ag][ak][al][be][bf][bh][bl][cd][ce][ch][cl][cm][cn][cr][dd][df][dh][di][dn][do][dp][ee][eg][ei][ej][ek][eo][eq][er][es][fc][fd][fe][ff][fg][fi][fj][fr][gb][ge][gp][ha][hb][hf][hg][hp][hq][hs][ia][ib][ig][iq][ir][is][ja][jc][jf][jg][jh][jr][js][kc][ke][kg][kh][kq][ks][lc][ld][lf][lg][lo][lp][lq][lr][md][mq][nf][nq][oa][or][pa][pb][pr][qc][qd][qe][qf][qq][qr][rb][rd][rp][rr][sd][se][sn][so][sp][sq]",
    "AB[ab][bb][bc][cc][cd][dc][eb][fa][gb][gc][hc][hd][gd][id][he][hf][gf][ff][ed][ee][fg][gh][hh][ig][jg][kg][kf][ke][kd][lf][mf][me][md][lg][lh][kh][mh][ji][jj][kj][lj][li][ni][nj][mj][jk][jl][lk][ll][ml][nl][nk][ok][ol][oj][pj][qj][qi][qh][qk][rk][ql][sl][rm][rn][qo][qp][qq][rq][rr][sr][pg][pf][of][qe][qd][rd][sd][se][qc][pb][pa][oa][na][qa][rb][or][mr][jn][jo][kp][kq][jq][jr][ir][is][io][ho][go][fp][fn][en][eo][dn][cn][dp][dq][dr][cr][cs][bs][em][fm][gm][el][ek][ej][fk][dk][dj][ci][cj][ck][cl][ah][ai][bi][bj][ln][lo][mo][mn][mp][no]AW[ac][ad][bd][be][ce][de][dd][df][ef][eg][eh][fh][gi][hi][ii][ih][jh][ij][ik][il][jm][km][kl][kk][kn][ko][lm][mm][nm][nn][om][on][pm][pn][qm][qn][pl][pk][po][pp][pq][oq][np][lp][lq][mq][lr][kr][ks][js][nr][pr][qr][qs][rs][ss][ag][bh][ch][cg][di][ei][fj][gj][gk][gl][fl][hm][hn][gn][in][iq][as][ar][br][cq][cp][bp][co][do][ao][bn][cm][bm][dm][dl][bl][bk][ak][aj][ga][ha][hb][ia][ic][jc][jb][jd][je][ie][if][jf][kc][lc][le][ld][mc][ma][nb][ob][pc][pd][pe][nd][ne][nf][mg][ng][nh][oh][oi][pi][ph][qf][qg][rg][rf][re][sf][rh][ri][rj][sj][sk]",
    "AB[af][ag][ap][aq][bf][bq][ca][cb][cd][ce][cf][cg][co][cp][cq][db][dc][dd][df][dn][do][dp][ef][eg][en][ep][fc][fd][fe][fg][fm][fn][fq][gd][gg][gp][hd][hg][hl][ho][hp][hq][id][ie][if][ig][ih][il][in][ip][iq][ir][jc][jd][je][jg][jj][jk][jn][jr][js][kd][ke][kg][ki][kj][kn][ko][kp][kq][ks][lh][li][lm][ln][lo][mb][md][me][mh][mj][mk][mn][mo][nb][nc][nh][nj][nm][nn][ob][oc][og][oi][om][pa][pc][pg][ph][pi][pj][po][pp][qb][qc][qf][qg][qk][ql][qm][qn][qo][qp][ra][rc][re][rf][rg][rh][ri][rn][sb][si][sn][so]AW[ah][ai][am][ao][bg][bh][bi][bl][bn][bo][bp][ch][ci][ck][cn][da][de][dg][dh][dk][dl][dm][ea][eb][ec][ed][ee][eh][em][eo][fb][fh][fl][fo][fp][ga][gc][gh][gl][gm][gn][go][hb][hc][hh][hm][hn][ic][ii][ij][ik][im][jb][jf][jh][ji][jl][jm][kc][kf][kh][kk][kl][km][kr][la][lb][lc][ld][le][lf][lg][lj][lk][ll][lp][lq][lr][ls][ma][mc][mg][ml][mm][mp][mr][na][nd][ng][nk][nl][no][oa][od][of][oj][ok][ol][on][oo][op][pd][pf][pk][pl][pm][pn][pq][qd][qe][qh][qi][qq][rd][ro][rp][rq][sc][sd][se][sf][sg][sh][sp]",
    "AW[ad][al][am][an][ao][ap][aq][bc][bd][be][bp][cc][ck][cl][co][cp][dc][df][dg][dh][di][dn][dp][ed][ee][ef][eh][ei][ej][en][ep][es][fe][fh][fi][fm][fp][fs][gb][gd][gg][gh][gm][gq][gs][ha][hb][hc][hd][hh][hi][hj][hm][hn][ho][hp][hq][hr][hs][ib][ig][ih][ii][il][in][iq][jg][jh][jk][jl][jn][jr][kg][ki][kj][kk][kq][lf][lg][ll][lq][lr][me][mf][mg][mh][ml][mq][ms][nd][ne][nh][nl][nr][oe][og][ol][oo][op][oq][or][os][pg][ph][pi][pl][pm][po][ps][qf][qi][qk][ql][rd][re][rf][rh][rk][rl][rm][se][sg][sk][sl]AB[ae][af][ak][ar][bf][bk][bl][bm][bn][bo][bq][bs][cd][ce][cf][cg][ch][ci][cm][cn][cq][cr][dd][de][dj][dk][dl][dm][dq][ds][eg][ek][em][eq][er][ff][fg][fj][fl][fq][fr][ge][gf][gi][gj][gl][go][gp][gr][he][hf][hg][hk][hl][ia][ic][id][ie][if][ij][ik][im][io][ip][ja][jb][jf][ji][jj][jm][jo][jp][jq][kb][kc][kd][kf][kh][kl][km][kn][ko][kp][lc][ld][le][lh][lm][lp][md][mm][mp][nc][nf][ng][nm][no][np][nq][od][of][om][on][pd][pe][pf][pn][pp][pq][pr][qc][qd][qe][qm][qn][qo][qp][qr][qs][rc][rn][sc][sd][sm][sn]",
    "AB[bb][bc][cc][cf][cn][dd][de][dj][dq][ee][fc][fe][nc][nd][pp][qn]AW[cb][cd][cg][dc][dg][ec][ed][fd][fh][gd][jc][jq][nq][oc][pd][qf]",
    "AW[bq][ce][cj][co][cp][dd][dm][dn][do][dq][fc][gp][gq][hn][qf][qj]AB[br][cf][cn][cq][cr][df][dp][en][eo][ep][fq][jp][nc][nq][pd][qp]",
    "AB[bb][bc][cc][cf][cn][dd][de][dj][dq][ee][fc][fe][nc][nd][pp][qn]AW[cb][cd][cg][dc][dg][ec][ed][fd][fh][gd][jc][jq][nq][oc][pd][qf]",
    "AW[bq][ce][cj][co][cp][dd][dm][dn][do][dq][fc][gp][gq][hn][qf][qj]AB[br][cf][cn][cq][cr][df][dp][en][eo][ep][fq][jp][nc][nq][pd][qp]",
    "AW[bq][ce][cj][co][cp][dd][dm][dn][do][dq][fc][gp][gq][hn][qf][qj]AB[br][cf][cn][cq][cr][df][dp][en][eo][ep][fq][jp][nc][nq][pd][qp]"
]

hp.num_images = len(board_info)


def main():
    # # import original image data
    # image = cv2.imread('./images/1.jpg')

    # # Resize image so it can be processed. Choose optimal dimensions such that important content is not lost
    # image = cv2.resize(image, (760, 760))
    # orig = image.copy()
    
    # plt.figure(0)
    # plt.imshow(image)
    # plt.show()

    # # Step 1: Edge Detection

    # # 1.1: Convert to grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # plt.figure(1, figsize=(7,7))
    # plt.imshow(gray, cmap='gray')

    # # 1.2: Blurring for Smoothness: Options-> Gaussian Blur, Median Blur
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # #blurred = cv2.medianBlur(gray, 5)
    # plt.figure(2, figsize=(7,7))
    # plt.imshow(blurred, cmap='gray')

    # # 1.3: Applying Canny Edge Detection
    # edged = cv2.Canny(blurred, 0, 50)
    # plt.figure(3, figsize=(7,7))
    # plt.imshow(edged, cmap='gray')

    # plt.show()

    # # Step 2: Finding largest contour in Edged Image

    # # 2.1: Find Contours
    # # (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # (contours, _) = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    # # 2.2 Sort contours by area in decreasing order
    # contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # # Plotting a bounding rectangle around largest contour for representational purposes
    # x,y,w,h = cv2.boundingRect(contours[0])
    # cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    # plt.figure(4, figsize=(7,7))
    # plt.imshow(image, cmap='gray')
    # plt.show()

    # # 2.3 Get largest approximate contour with 4 vertices
    # for c in contours:
    #     p = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, 0.02 * p, True)

    #     if len(approx) == 4:
    #         target = approx
    #         break

    # print('Largest approximate Contour is: ' + str(target))

    # # Plotting the largest contour for representational purposes
    # cv2.drawContours(image, [target], -1, (255, 0, 0), 2)
    # plt.figure(5, figsize=(7,7))
    # plt.imshow(image, cmap='gray')
    # plt.show()

    image = cv2.imread('./images/24.jpg')

    # Resize image so it can be processed. Choose optimal dimensions such that important content is not lost
    image = cv2.resize(image, (760, 760))
    orig = image.copy()
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # data = naive_stone_detection(image)
    data = stone_detection_with_hough_circles(image)
    


    # store the expected output
    for i in range(hp.num_images):
        # print(i)
        expected.append(convert_board_info_to_array(board_info[i]))

    # use pygame to visualize board
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        
        draw_board(data)

        pygame.display.update()
    
    pygame.quit()


def draw_board(board):
    window.fill(hp.GREY)
    for row in range(hp.ROW + 1):
        for col in range(hp.COL + 1):
            if row != hp.ROW and col != hp.COL:
                pygame.draw.rect(
                    window, 
                    hp.BACKGROUND_COLOR, 
                    (
                        (row + 1) * (hp.GRID_SIZE + 1), 
                        (col + 1) * (hp.GRID_SIZE + 1),
                        hp.GRID_SIZE,
                        hp.GRID_SIZE
                    )
                )
            ## black
            if board[col][row] == 1:
                pygame.draw.circle(
                window, 
                hp.BLACK, 
                (
                    (row + 1) * (hp.GRID_SIZE + 1), 
                    (col + 1) * (hp.GRID_SIZE + 1)
                ),
                hp.STONE_SIZE
            )

            ## white
            if board[col][row] == 2:
                pygame.draw.circle(
                window, 
                hp.WHITE, 
                (
                    (row + 1) * (hp.GRID_SIZE + 1), 
                    (col + 1) * (hp.GRID_SIZE + 1)
                ),
                hp.STONE_SIZE
            )

def convert_board_info_to_array(data):
    # initialize with zeroes
    board = [[0 for i in range(19)] for j in range(19)]

    # is of the pattern "[cd][ab]...", where cd means row d col c
    if data.index("AW") < data.index("AB"):
        white_substring = data.split("AW")[1].split("AB")[0]
        black_substring = data.split("AB")[1]
    else:
        white_substring = data.split("AW")[1]
        black_substring = data.split("AB")[1].split("AW")[0]

    for i in range(len(white_substring) // 4):
        row = ord(white_substring[i * 4 + 2]) - ord('a')
        col = ord(white_substring[i * 4 + 1]) - ord('a')
        # print("white: " + white_substring[i * 4 + 2] + white_substring[i * 4 + 1] + str(row) + ", " + str(col))
        board[row][col] = 2

    for i in range(len(black_substring) // 4):
        row = ord(black_substring[i * 4 + 2]) - ord('a')
        col = ord(black_substring[i * 4 + 1]) - ord('a')
        # print("black: " + black_substring[i * 4 + 2] + black_substring[i * 4 + 1] + str(row) + ", " + str(col))
        board[row][col] = 1
    
    return board


def naive_stone_detection(image):
    data = [[0 for i in range(19)] for j in range(19)]

    for i in range(19):
        for j in range(19):
            data[i][j] = naive_stone_color(image, i, j)
    
    return data

def naive_stone_color(image, i, j):
    row1 = i * 40 + 10
    col1 = j * 40 + 10
    row2 = i * 40 + 15
    col2 = j * 40 + 15
    row3 = i * 40 + 25
    col3 = j * 40 + 25
    row4 = i * 40 + 30
    col4 = j * 40 + 30
    if mean(image[row1][col1]) < 50 or mean(image[row2][col2]) < 50 or mean(image[row3][col3]) < 50 or mean(image[row4][col4]) < 50:
        # black
        return 1
    elif mean(image[row1][col1]) > 200 or mean(image[row2][col2]) > 200 or mean(image[row3][col3]) > 200 or mean(image[row4][col4]) > 200:
        # white
        return 2
    return 0

def stone_detection_with_hough_circles(image):
    data = [[0 for i in range(19)] for j in range(19)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        image=gray,
        method=cv2.HOUGH_GRADIENT,
        dp=1, minDist=10,
        param1=100, param2=25,
        minRadius=5, maxRadius=25
    )[0]
    circles = np.uint16(np.around(circles))

    for circle in circles:
        x, y, r = circle
        data[y // 40][x // 40] = stone_color_with_circle(image, x, y, r)

    return data

# reference: https://stackoverflow.com/questions/43086715/rgb-average-of-circles
def stone_color_with_circle(image, x, y, r):
    circle_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.circle(circle_img,(x, y),r,(255,255,255),-1)
    datos_rgb = cv2.mean(image, mask=circle_img)[::-1][1:]
    
    if mean(datos_rgb) < 100:
        return 1
    elif mean(datos_rgb) > 150:
        return 2
    return 0

main()