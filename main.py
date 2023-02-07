from data_generate import load_data
from train import Model

if __name__ == '__main__':
    supply, demand = load_data()
    m_rand = Model("Rand", supply, demand)
    m_rand.train()
    print("Result of Rand:")
    m_rand.delivery()

    m_pdb = Model("PBD", supply, demand)
    m_pdb.train()
    print("Result of PBD:")
    m_pdb.delivery()

    m_rand = Model("FACC", supply, demand)
    m_rand.train()
    print("Result of FACC:")
    m_rand.delivery()
