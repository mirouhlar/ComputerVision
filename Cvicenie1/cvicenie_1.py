import numpy as np
import time


def myfunction(x): # funkcia s jedným argumentom a jedným návratovým parametrom
    return np.array([-2, -1, 0, 1]) + x

def myotherfunction(x, y): # funkcia s dvoma argumentmi a dvoma návratovými parametrami
    return x + y, x - y


if __name__ == '__main__':
    print("################################")
    A = np.array([[1, 2], [3, 4]])  # Vyvorenie matice 2x2
    B = np.array([[1, 2], [3, 4]])  # Vyvorenie matice 2x2
    N = 5  # Skalár
    v_1 = np.array([[1, 2, 3]])  # Riadkový vektor
    v_2 = np.array([[1], [2], [3]])  # Stĺpcový vektor
    v_3 = v_2.T  # Transpozícia vektora/matice
    v_4 = np.arange(1, 3, 0.5)  # Vektor vytvorený číslami určitého rozsahu
    v_5 = np.array([])  # Prázdny vektor
    print(
        "Matica A:\n {}\nMatica B:\n {}\nSkalár N:\n {}\nRiadkový vektor \n {}\nStĺpcový vektor\n {}\nTransponovaná matica \n {}\nVektor tvorený číslami z rozsahu\n {}\n".format(
            A, B, N, v_1, v_2, v_3, v_4))
    print("################################")

    ################################

    m_1 = np.zeros((2, 3))  # Matica romzmeru 2x3 tvorená nulami
    m_2 = np.ones((1, 3))  # Matica rozmeru 1x3 tvorená jednotkami
    m_3 = np.eye(3)  # Jednotková matica
    m_4 = np.random.rand(3, 3)  # Matica rozmeru 3x1 náhodných čísiel
    m_5 = np.zeros((3, 3))  # Matica rozmeru 3x3 tvorená nulami
    print(
        "Nulová matica \n {}\nVektor tvorený jednotkami \n {}\nJednotková matica \n {}\nMatica náhodných čísiel\n {}".format(
            m_1, m_2, m_3, m_4))

    print("################################")
    ################################

    n_1 = np.array([1, 2, 3])  # Vektor
    print("\nVektor n_1:", n_1)
    print("\nTreti prvok vektora je:\t", n_1[2])
    n_2 = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]])  # Matica
    print("\nMatica n_2:\n", n_2)
    print("\nDruhy prvok tretieho riadka je:\t", n_2[2][1])
    print("\nPrvý riadok matice n_2: \n", n_2[[0], :])
    print("\nPrvý stĺpec matice n_2: \n", n_2[:, [0]])
    print("\nPrvky matice n_2 na indexoch [1][1-3]: \n", n_2[[0], 0:3])
    print("\nPrvky matice n_2 na indexoch [1-3][1]: \n", n_2[0:3, [0]])
    print("\nPrvky matice n_2 na indexoch [2-koniec][3]: \n", n_2[1:, [2]])

    o = np.array([[1, 2, 3], [4, 5, 6]])
    print("\nMatica o: \n", o)
    print("\nRozmery matice o: \n", o.shape)
    print("\nPočet riadkov matice o: \n", o.shape[0])
    print("\nPočet stĺpcov matice o: \n", o.shape[1])
    o_1 = np.zeros(o.shape)
    print("\nNulová matica o_1 s rozmermi matice o: \n", o_1)
    print("################################")
    ################################

    a = np.array([[1, 2, 3, 4]]).T  # Stĺpcový vektor
    print("\nVektor a: \n", a)
    print("\nNásobenie matice a skalárom (2 * a): \n", 2 * a)
    print("\nDelenie matice a skalárom (a / 4): \n", a / 4)
    b = np.array([[5, 6, 7, 8]]).T  # Stĺpcový vektor
    print("\nVektor b: \n", b)
    print("\nSúčet vektorov (a + b)\n", a + b)
    print("\nRozdiel vektorov (a - b)\n", a - b)
    print("\nUmocnenie prvkov vektora (a ^ 2)\n", a ** 2)
    print("\nNásobenie matíc prvok po prvku (a * 2)\n", a * b)
    print("\nDelenie matíc prvok po prvku (a / 2)\n", a / b)
    print("\nLogaritmus vektora [1 2 3 4] prvok po prvku\n", np.log(np.array([[1, 2, 3, 4]])))
    print("\nZaokrúhlenie prvkov vektora [1.5 2.2 3.8 4.4] k najbližšiemu celému číslu\n",
          np.round(np.array([[1.5, 2.2, 3.8, 4.4]])))
    print("################################")
    ################################

    g = np.array([[1, 4, 6, 3]])  # Riadkový vektor
    print("\nRiadkový vektor g\n", g)
    print("\nSuma prvkov vektora g:\n", np.sum(g))  # Suma prvkov vektora
    print("\nPriemer prvkov vektora g:\n", np.mean(g))  # Priemer prvkov vektora
    print("\nRozptyl vektora g:\n", np.var(g))  # Rozptyl vektora
    print("\nSmerodajná odchýlka vektora g:\n", np.std(g))  # Smerodajná odchýlka vektora
    print("\nMaximum vektora g:\n", np.max(g))  # Maximum vektora
    print("\nMinimum vektora g:\n", np.min(g))  # Minimum vektora
    print("################################")
    ################################

    h = np.array([[1, 2, 3], [4, 5, 6]])  # Matica
    print("\nMatica h\n", h)
    print("\nPriemer každého stĺpca\n", np.mean(h, 0))  # Priemer
    print("\nPriemer každého riadku\n", np.mean(h, 1))  # Priemer každého riadku
    print("\nMaximum každého stĺpca \n", np.max(h, 0))  # Maximum každého stĺpca
    print("\nMaximum celej matice\n", np.max(h))  # Maximum z celej matice
    print("################################")
    ################################

    i = np.array([[1, 2, 3]])  # Vektor
    j = np.array([[4, 5, 6]])  # Vektor
    print("\nVektor i\n", i)
    print("\nVektor j\n", j)
    print("\nSkalárny súčin matíc i a j'\n", i @ j.T)  # Skalárny súčin i a j'
    print("\nVektorový súčin matíc i' a j\n", i.T * j)  # Vektorový súčin i' a j
    print("################################")
    ################################

    l_1 = np.random.rand(3, 2)  # Matica náhodných čísiel rozmeru 3x2
    l_2 = np.random.rand(2, 4)  # Matica náhodných čísiel rozmeru 2x4
    print("\nMatica l_1\n", l_1)
    print("\nMatica l_2\n", l_2)
    l_3 = l_1 @ l_2  # Skalárny súčin matíc produkujúci maticu 3x4
    print("\nSkalárny súčin matíc l_1 a l_2\n", l_3)
    l_1 = np.array([[1, 2], [3, 4], [5, 6]])  # Matica 3x2
    l_2 = np.array([[5, 6, 7]])  # Riadkový vektor 1x3
    print("\nMatica l_1\n", l_1)
    print("\nVektor l_2\n", l_2)
    print("\nSúčin vektora l_2 a matice l_1\n", np.matmul(l_2, l_1))

    l_4 = np.array([[8], [9]])  # 2x1 stĺpcový vektor
    print("\nStĺpcový vektor l_4\n", l_4)
    print("\nSúčin matice l_1 a vektora l_4\n", np.matmul(l_1, l_4))

    l = np.array([[1, 2, 3], [6, 50, 4], [70, 8, 90]])  # Matica 3x3
    print("\nMatica l\n", l)
    print("\nInverzná matica k matici l\n", np.linalg.inv(l))
    print("\nVektor vlastných hodnôt matice l\n", np.linalg.eigvals(l))
    V, D = np.linalg.eig(l)  # Matica vlastných vektorov V a matica s vlastnými hodnotami na diagonále D
    print("\nMatica vlastných vektorov V matice l\n {}\nMatica s vlastnými hodnotami na diagonále\n {}\n".format(V, D))
    U, S, V = np.linalg.svd(l)  # Rozklad matice na singulárne body SVD l = U * S * V'
    print("\nRozklad matice l na singulárne body S V D\nU {}\nS {}\nV {}\n ".format(S, V, D))
    print("################################")
    ################################

    a = np.array([[1, 2], [3, 4], [5, 6]])  # Matica rozmeru 3x2
    print("\nMatica a\n", a)
    b = a.flatten('F')  # Vektor 6x1 vytvorený uloženík stĺpcov matice a na seba
    print("\nVektor b vytvorený uloženík stĺpcov matice a na seba\n", b)
    print("\nSuma všetkých prvkov vektora b\n", np.sum(b))
    print("\nZmena rozmeru matice b na rozmer 2x3\n", b.reshape(2, 3))  # Vytvorí z vektora v maticu rozmeru 2x3

    a = np.array([[1, 2]])  # Riadkový vektor
    b = np.array([[3, 4]])  # Riadkový vektor
    c = np.block([a, b])  # Vodorovné spájanie vektorov
    print("\nVodorovné spájanie vektorov\n", c)

    a = np.array([[1], [2], [3]])  # Stĺpcový vektor
    print("\nZvislé spájanie vektorov\n", np.block([[a], [4]]))

    a = np.block([np.eye(3), np.random.rand(3, 3)])  # Spájanie matíc - vedla seba
    print("\n", a)
    a = np.block([[np.eye(3)], [np.ones((1, 3))]])  # Spájanie matíc - na seba
    print("\n", a)

    b = np.tile(5, (3, 2))  # Matica príslušného rozmeru, ktorej prvky sú 5
    print("\n", b)
    b = np.tile(np.array([[1, 2], [3, 4]]), (1, 2))  # Replikuje maticu 2x2 dvakrát v smere stĺpcov
    print("\n", b)
    b = np.diag([1, 2, 3])  # Matica rozmery 3x3 s príslušnými prvkami na hlavnej diagonále
    print("\n", b)
    print("################################")
    ################################

    for i in range(1, 7, 2):
        print(i, " ")

    for i in np.array([5, 13, -1]):
        if i > 10:
            print("Larger than 10\n")
        elif i < 10:
            print("Negative value\n")
        else:
            print("Something else\n")

    print("################################")
    ################################

    m = 50
    n = 10
    A = np.ones((m, n))
    v = 2 * np.random.rand(1, n)

    # implementácia využitím cyklu
    for i in range(1, m, 1):
        A[i][:] = A[i][:] - v
    print(A)

    # implementácia využitím maticových operácií
    A = np.ones((m, n)) - np.tile(v, (m, 1))
    print(A)

    # implementácia využitím cyklu
    B = np.zeros([m, n])
    for i in range(0, m):
        for j in range(1, n):
            if A[i][j] > 0:
                B[i][j] = A[i][j]
    print(B)
    # implementácia bez cyklu
    B = np.zeros([m, n])
    ind = np.where(A > 0)
    B[ind] = A[ind]
    print(B)

    print("################################")
    ################################

    t_0 = time.time()
    x = np.empty([1, 1000000])
    for i in range(1, 1000000):
        x[0][i] = x[0][i - 1] + 5
    t_1 = time.time()
    print("Cas bez alokacie:\t", t_1 - t_0)

    t_0 = time.time()
    x = np.zeros([1, 1000000])
    for i in range(1, 1000000):
        x[0][i] = x[0][i - 1] + 5
    t_1 = time.time()
    print("Alokácia:\t", t_1 - t_0)

    # zoznam
    t_0 = time.time()
    T = [0]
    for i in range(1, 1000000):
        T.append((T[i - 1] + 5))
    X = np.array(T)
    t_1 = time.time()
    print("Alokácia - využitie zoznamu:\t", t_1 - t_0)
    print("################################")
    ################################

    a = np.array([[1, 2, 3, 4]])  # vektor
    b = myfunction(2 * a)  # volanie vlastnej funkcie s nazvom myfunction
    print("b\n", b)

    c, d = myotherfunction(a, b)  # volanie vlastnej funkcie s nazvom mymotherfunction
    print("c\n {}\nd\n {}".format(c, d))
