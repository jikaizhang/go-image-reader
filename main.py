import hyperparameters as hp
import pygame
import cv2
import matplotlib.pyplot as plt
import numpy as np
from statistics import mean
import tkinter as tk
from tkinter import simpledialog
import easygui



window = pygame.display.set_mode((hp.WIDTH, hp.HEIGHT))
pygame.display.set_caption('Go Reader')

# 0: empty
# 1: black
# 2: white
expected = []

board_info = [
    "AW[ae][af][ag][ai][al][an][bd][be][bg][bi][bj][bm][bo][cj][cn][co][di][dl][dm][dn][ed][eh][ei][el][en][eo][ep][fi][fj][fk][fq][gf][gh][gi][gk][gn][go][gq][gr][gs][hc][hg][hh][hi][hj][ho][hq][ic][id][ie][ii][il][im][ip][iq][jb][jc][jf][jm][jn][jo][jp][kc][kf][kg][kh][ki][kj][km][ko][lc][lg][li][lk][ll][lm][lr][ls][mc][md][mg][mi][mk][mn][ms][na][nb][ne][nf][ni][nj][nl][nm][nn][no][np][nq][nr][ns][oc][of][oh][ok][op][pb][pc][pd][pf][ph][pi][pj][pk][pp][qc][qe][qf][qh][qj][ql][qp][qq][qr][rd][rj][rq][sd]AB[ah][ao][ap][bc][bf][bh][bp][cd][ce][cg][ch][ci][cp][dc][dd][dh][do][dp][dq][dr][ee][ef][eg][em][eq][es][fb][fc][fe][fg][fh][fl][fm][fn][fo][fr][fs][ga][gc][ge][gg][gl][ha][hb][hd][he][hf][hk][hl][hm][hn][hp][ia][ib][if][ig][ih][ij][ik][in][io][ir][is][ja][jg][jh][ji][jj][jl][jq][jr][kb][kk][kn][kp][kr][ks][la][lb][lh][ln][lo][lq][mb][mh][mo][mp][mq][mr][ng][nh][ob][og][ol][oo][pg][pl][pm][po][qg][qi][qk][qn][qo][re][rf][rg][rh][ri][rk][rl][rn][rp][rr][se][sg][sh][sj][sl][sp][sq][ss]",
    "AW[ac][af][aj][ak][bb][bd][be][bf][bg][bi][bj][bl][bm][br][bs][ca][cc][ce][cf][ck][cl][cm][co][cq][cr][da][db][dc][de][df][dn][do][dp][ea][ee][ef][eg][eh][ei][ej][eo][ep][eq][fe][ff][fl][fo][fp][gd][ge][gf][gk][gl][gm][gn][go][hc][hf][hg][hk][hn][hp][ic][ie][il][im][in][jc][je][jf][jm][jr][js][kb][kd][kg][ki][km][kn][ld][lg][lh][li][lj][lk][lm][ln][lr][ls][mc][md][me][mf][mj][mq][ms][nd][ng][nh][ni][nj][nk][nn][nq][nr][oc][od][oe][of][oj][oq][or][os][pd][pf][pg][pn][pq][pr][ps][qe][qf][qg][qh][qi][qj][ql][qn][qp][qq][qr][rf][rh][rk][rl][rm][rn][rp][sh][sm]AB[ag][ah][ai][am][ba][bh][bn][cd][cg][ch][ci][cj][cn][cs][dd][dg][dh][di][dj][dk][dl][dm][dq][dr][ds][eb][ec][ed][ek][el][em][en][er][es][fa][fb][fd][fg][fh][fi][fj][fk][fm][fn][fq][fs][ga][gc][gg][gj][gp][gq][gr][gs][hb][hd][he][hh][hi][hj][ho][ib][if][ig][ih][ij][ik][io][jb][jg][jh][ji][jk][jl][jn][jo][jq][kc][kf][kh][kj][kk][kl][ko][kp][kq][kr][ks][lb][lc][lf][ll][lo][lp][lq][mb][mi][mk][ml][mm][mn][mp][nb][nc][nf][nl][nm][np][ob][og][oh][oi][ok][on][oo][op][pc][ph][pi][pj][pk][pl][pm][po][pp][qb][qc][qd][qk][qm][qo][qs][rc][re][rg][ro][rq][rr][rs][sd][se][sf][sg][sn][so][sp]",
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

num_clicks = 0
board_corner_positions = [[0 for i in range(2)] for j in range(4)]

def main():
    # Some popups to understand user's preferences
    root = tk.Tk()
    root.withdraw()

    mode = " "
    image_path = " "
    select_original_image = " "

    while mode != "a" and mode != "b":
        mode = simpledialog.askstring(title="Choose Mode",
                prompt="(A) Upload your own image\n(B) Use our images")
        mode = mode.lower()

    # user upload
    if mode == "a":
        while image_path == " ":
            image_path = simpledialog.askstring(title="Provide Image Path",
                prompt="Please provide your image path:")
    # our images
    else:
        while select_original_image != "a" and select_original_image != "b":
            select_original_image = simpledialog.askstring(title="Choose Folder",
                    prompt="(A) Use original images\n(B) Use preprocessed images")
            select_original_image = select_original_image.lower()
        if select_original_image == "a":
            image_path = "./images"
        else:
            image_path = "./nice_images"


    def click_event(event, x, y, flags, params):
        global num_clicks

        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(orig, (x, y), 10, (255, 0, 0), -1)
            cv2.imshow('click to select the four corners of the board', orig)

            board_corner_positions[num_clicks][0] = x
            board_corner_positions[num_clicks][1] = y

            num_clicks += 1

            if num_clicks == 4:
                num_clicks = 0
                cv2.destroyAllWindows()
    
    # user can only upload one image
    if mode == "a":
        num_images = 1
    # or we go through all images in our folder
    else:
        num_images = hp.num_images

    # total number of points on all boards
    total_points = 0
    # number of stones we recognized correctly (regardless of colors)
    correct_circles = 0
    # number of stones we labelled correctly (is it a stone? black or white?)
    correct_points = 0

    # show the accuracy
    for i in range(num_images):
        # this is the expected output
        expected_board = convert_board_info_to_array(board_info[i])
        expected.append(expected_board)

        if mode == "a":
            image = cv2.imread(image_path)
        else:
            image = cv2.imread('{}/{}.jpg'.format(image_path, i + 1))

        # total number of points on current board
        curr_total_points = 0
        # number of stones we recognized correctly (regardless of colors)
        curr_correct_circles = 0
        # number of stones we labelled correctly (is it a stone? black or white?)
        curr_correct_points = 0

        # Resize image so it can be processed. Choose optimal dimensions such that important content is not lost
        image = cv2.resize(image, (hp.converted_image_size, hp.converted_image_size))

        copy = image.copy()

        # reduce the effect of bright reflection on image
        image = reduce_bright_reflection(image)
        orig = image.copy()

        # we need to ask the user to select the four corners
        # and crop the board
        if mode == "a" or select_original_image == "a":
            easygui.msgbox("Please click on the four corners of the board on the following image!", title="Alert")

            # ask the user to select the four corners of the board
            cv2.imshow('click to select the four corners of the board', orig)

            # setting mouse handler for the image
            # and calling the click_event() function
            cv2.setMouseCallback('click to select the four corners of the board', click_event)
        
            # wait for a key to be pressed to exit
            cv2.waitKey(0)

            # close the window
            cv2.destroyAllWindows()

            # reference: https://stackoverflow.com/questions/42262198/4-point-persective-transform-failure
            # now we got the four corners, let's crop the board

            # rearrage the user clicked 4 corners in clockwise order
            src_pts = np.array(rearrage_points_clockwise(board_corner_positions), dtype=np.float32)
            width = get_euler_distance(src_pts[0], src_pts[1])
            height = get_euler_distance(src_pts[0], src_pts[3])

            dst_pts = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32)

            # transform the cropped board to a flat image
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warp = cv2.warpPerspective(image, M, (int(width), int(height)))

            plt.figure('warp', figsize=(7,7))
            plt.imshow(warp)
            plt.show()

            image = cv2.resize(warp, (hp.converted_image_size, hp.converted_image_size))

        # real_board = naive_stone_detection(image)
        real_board = stone_detection_with_hough_circles(image)

        # if the user choose to use our images, show (s)he the accuracy
        if mode == "b":
            for j in range(19):
                for k in range(19):
                    curr_total_points += 1
                    if (expected_board[j][k] == 0 and real_board[j][k] == 0) or (expected_board[j][k] != 0 and real_board[j][k] != 0):
                        curr_correct_circles += 1
                    if expected_board[j][k] == real_board[j][k]:
                        curr_correct_points += 1

            print("Image #{}".format(i + 1))
            print("Accuracy of detecting a stone or a none stone is: {}".format(curr_correct_circles / curr_total_points))
            print("Accuracy of detecting a stone and its color or a none stone is: {}".format(curr_correct_points / curr_total_points))
            print()
            total_points += curr_total_points
            correct_circles += curr_correct_circles
            correct_points += curr_correct_points

        if mode == "a":
            # use pygame to visualize board
            run = True

            while run:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                
                # visualize the model's labels in a customized GUI
                draw_board(real_board)

                pygame.display.update()
            
            # Show the original image
            plt.figure('original image', figsize=(7,7))
            plt.imshow(copy)
            plt.show()

            pygame.quit()

    if mode == "b":
        print("Overall accuracy of detecting a stone or a none stone is: {}".format(correct_circles / total_points))
        print("Overall accuracy of detecting a stone and its color or a none stone is: {}".format(correct_points / total_points))


# visualize the 2d board array with my customized GUI
def draw_board(board):
    window.fill(hp.GREY)
    for row in range(hp.ROW + 1):
        for col in range(hp.COL + 1):
            # draw a grid
            if row != hp.ROW and col != hp.COL:
                pygame.draw.rect(
                    window, 
                    hp.BACKGROUND_COLOR, 
                    (   
                        # hp.GRID_SIZE + 1 so we have 1px lines in the board
                        (row + 1) * (hp.GRID_SIZE + 1), 
                        (col + 1) * (hp.GRID_SIZE + 1),
                        hp.GRID_SIZE,
                        hp.GRID_SIZE
                    )
                )
            ## black stone
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

            ## white stone
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


# convert the sgf info and return the converted info
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
        board[row][col] = 2

    for i in range(len(black_substring) // 4):
        row = ord(black_substring[i * 4 + 2]) - ord('a')
        col = ord(black_substring[i * 4 + 1]) - ord('a')
        board[row][col] = 1
    
    return board


# a naive implementation of stone detection, where we simply
# classify the color in a given grid
def naive_stone_detection(image):
    data = [[0 for i in range(19)] for j in range(19)]

    for i in range(19):
        for j in range(19):
            data[i][j] = naive_stone_color(image, i, j)
    
    return data


# a naive way to determine the stone color: pick four points, and
# check if there is a point that's very bright (white) or very
# dark (black), otherwise classify it as empty space
def naive_stone_color(image, i, j):
    row1 = int((i + 0.25) * hp.converted_grid_size)
    col1 = int((j + 0.25) * hp.converted_grid_size)
    row2 = int((i + 0.375) * hp.converted_grid_size)
    col2 = int((j + 0.375) * hp.converted_grid_size)
    row3 = int((i + 0.625) * hp.converted_grid_size)
    col3 = int((j + 0.625) * hp.converted_grid_size)
    row4 = int((i + 0.75) * hp.converted_grid_size)
    col4 = int((j + 0.75) * hp.converted_grid_size)
    if mean(image[row1][col1]) < 50 or mean(image[row2][col2]) < 50 or mean(image[row3][col3]) < 50 or mean(image[row4][col4]) < 50:
        # black
        return 1
    elif mean(image[row1][col1]) > 200 or mean(image[row2][col2]) > 200 or mean(image[row3][col3]) > 200 or mean(image[row4][col4]) > 200:
        # white
        return 2
    return 0


# using openCV's HoughCircles library to find all stones (circles)
# and then determine the stone color
def stone_detection_with_hough_circles(image):
    data = [[0 for i in range(19)] for j in range(19)]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    circles = cv2.HoughCircles(
        image=gray,
        method=cv2.HOUGH_GRADIENT,
        dp=1, minDist=15,
        param1=100, param2=25,
        minRadius=2, maxRadius=45
    )[0]
    circles = np.uint16(np.around(circles))
    
    # compute the average stone color, anything color darker than it 
    # would be considered black, else white
    avg_stone_color = average_stone_color(image, circles)

    for circle in circles:
        x, y, r = circle
        # use a smaller radius to reduce the effect of glare on stones
        # or irregular circles
        data[y // hp.converted_grid_size][x // hp.converted_grid_size] = stone_color_with_circle(image, x, y, int(r * 0.4), avg_stone_color)
        
    return data


# compute the average pixel of all stones in the board, 
# anything color darker than it would be considered black, else white
def average_stone_color(image, circles):
    color_sum = 0
    for circle in circles:
        x, y, r = circle
        circle_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
        cv2.circle(circle_img,(x, y),r,(255,255,255),-1)
        datos_rgb = cv2.mean(image, mask=circle_img)[::-1][1:]
        color_sum += mean(datos_rgb)
    
    return color_sum / len(circles)


# determine the color of the stone by comparing it with the 
# average pixels of all stones on the board
# reference: https://stackoverflow.com/questions/43086715/rgb-average-of-circles
def stone_color_with_circle(image, x, y, r, avg_stone_color):
    circle_img = np.zeros((image.shape[0], image.shape[1]), np.uint8)
    cv2.circle(circle_img,(x, y), r, (255, 255, 255), -1)
    datos_rgb = cv2.mean(image, mask=circle_img)[::-1][1:]
    
    if mean(datos_rgb) < avg_stone_color:
        return 1
    return 2


# because the model did not perform well when there is strong light spot
# on the board, I try to reduce that influence
# reference: https://stackoverflow.com/questions/43470569/remove-glare-from-photo-opencv
def reduce_bright_reflection(img):
    clahefilter = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16,16))

    # NORMAL
    # convert to gray
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayimg = gray

    GLARE_MIN = np.array([0, 0, 50],np.uint8)
    GLARE_MAX = np.array([0, 0, 225],np.uint8)

    hsv_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

    # HSV
    frame_threshed = cv2.inRange(hsv_img, GLARE_MIN, GLARE_MAX)

    # INPAINT
    mask1 = cv2.threshold(grayimg , 220, 255, cv2.THRESH_BINARY)[1]
    result1 = cv2.inpaint(img, mask1, 0.1, cv2.INPAINT_TELEA) 

    # CLAHE
    claheCorrecttedFrame = clahefilter.apply(grayimg)

    # COLOR 
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab_planes = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes[0] = clahe.apply(lab_planes[0])
    lab = cv2.merge(lab_planes)
    clahe_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # INPAINT + HSV
    result = cv2.inpaint(img, frame_threshed, 0.1, cv2.INPAINT_TELEA) 

    # INPAINT + CLAHE
    grayimg1 = cv2.cvtColor(clahe_bgr, cv2.COLOR_BGR2GRAY)
    mask2 = cv2.threshold(grayimg1 , 220, 255, cv2.THRESH_BINARY)[1]
    result2 = cv2.inpaint(img, mask2, 0.1, cv2.INPAINT_TELEA) 

    # HSV + INPAINT + CLAHE
    lab1 = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
    lab_planes1 = cv2.split(lab1)
    clahe1 = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    lab_planes1[0] = clahe1.apply(lab_planes1[0])
    lab1 = cv2.merge(lab_planes1)
    clahe_bgr1 = cv2.cvtColor(lab1, cv2.COLOR_LAB2BGR)

    cv2.imshow("HSV + INPAINT + CLAHE   ", clahe_bgr1)

    return clahe_bgr1


# get the euler distance between two points, used to get the cropped
# area on an image given the four corners
def get_euler_distance(pt1, pt2):
    return ((pt1[0] - pt2[0])**2 + (pt1[1] - pt2[1])**2)**0.5


# return the clockwise order of a list of points
def rearrage_points_clockwise(positions):
    mean_x = mean([row[0] for row in positions])
    mean_y = mean([row[1] for row in positions])

    res = [[0 for i in range(len(positions[0]))] for j in range(len(positions))]

    # the order is:
        # 0 = top-left
        # 1 = top-right
        # 2 = bottom-right
        # 3 = bottom-left
    for point in positions:
        if point[0] < mean_x and point[1] < mean_y:
            res[0] = point
        elif point[0] > mean_x and point[1] < mean_y:
            res[1] = point
        elif point[0] > mean_x and point[1] > mean_y:
            res[2] = point
        else:
            res[3] = point
    
    return res


main()