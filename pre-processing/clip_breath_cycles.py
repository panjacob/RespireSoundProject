from tqwt_stft import clip_cycle
if __name__ == "__main__":
    dir = "../data/official/test/"
    newdir = "../data/breath_cycles/test/"
    clip_cycle(dir, newdir)