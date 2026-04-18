from data.bonds import get_all_bonds_data


def main():
    print(get_all_bonds_data()["instruments"][0]["ticker"])


if __name__ == "__main__":
    main()
