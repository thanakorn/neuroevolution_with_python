import os
import shutil
import neat
import visualize

local_dir = os.path.dirname(__file__)
out_dir = os.path.join(local_dir, 'out')

def eval_fitness(net):
    xor_inputs = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
    xor_outputs = [0.0, 1.0, 1.0, 0.0]
    error = 0.0
    for (xi, xo) in zip(xor_inputs, xor_outputs):
        error += abs(xo - net.activate(xi)[0])
    return (4 - error) ** 2

def eval_genomes(genomes, config):
    for id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = eval_fitness(net)

def run_experiment(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_file)
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(show_species_detail=True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(generation_interval=5, filename_prefix=os.path.join(local_dir,'out/neat-checkpoint-')))
    best_genome = p.run(fitness_function=eval_genomes, n=300)

    best_net = neat.nn.FeedForwardNetwork.create(best_genome, config)
    best_genome_fitness = eval_fitness(best_net)
    if best_genome_fitness > config.fitness_threshold:
        print('\n\nSuccess: The XOR problem solver found!!!')
    else:
        print('\n\nFailure: Failed to find XOR problem solver!!!')

    node_names = {-1:'A', -2:'B', 0:'A XOR B'}
    visualize.draw_net(config, best_genome, True, node_names=node_names, directory=out_dir)
    visualize.plot_stats(stats, ylog=False, view=True, filename=os.path.join(out_dir, 'avg_fitness.svg'))
    visualize.plot_species(stats, view=True, filename=os.path.join(out_dir, 'speciation.svg'))

def clean_output():
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=False)

if __name__=='__main__':
    config_path = os.path.join(local_dir, 'config.ini')
    clean_output()
    run_experiment(config_path)
