def generate_movie(length, movie_name, start_after = 1):
    prepare_pymol()
    cmd.mset("1x"+str(30*length))
    cmd.zoom("all", buffer=42, state=-1)
    cmd.frame(1)
    cmd.mview("store", object='AA')
    cmd.frame(30*start_after) # start animation after start_after
    cmd.mview("store", object='AA')
    cmd.frame(30*2) # four seconds of translation
    cmd.translate(translation, object='AA')
    cmd.mview('store', object='AA')
    cmd.frame(30*2) # 4 seconds of rotation
    for i in range(3):
        cmd.rotate(axis[i], rotation[i], object="AA")
    cmd.mview('store', object='AA')
    cmd.frame(30*2)
    unfold()
    cmd.mview('store', object='AA')
    cmd.frame(30*length)
    cmd.mview('interpolate', object='AA')
    movie.produce(filename = movie_name+'.mpg')
