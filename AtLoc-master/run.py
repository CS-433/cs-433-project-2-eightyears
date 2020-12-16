from tools.options import Options

if __name__ == '__main__':
    opt = Options().parse()
    if opt.final_weights == '':
        import data.dataset_mean
    else:
        import tools.saliency_imgs

