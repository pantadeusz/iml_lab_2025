def main():
    name_and_surname = "Damian Raczynski"

    numbers = [12, 32, 44, 23, 53]

    for n in numbers:
        print(n, end=" ")
    print()

    print(f"Suma liczb 5 + 4 = {sum_of_two_numbers(5, 4)}")


def sum_of_two_numbers(a: int, b: int) -> int:
    return a + b


if __name__ == "__main__":
    main()
