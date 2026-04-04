import argparse
import os
import pickle

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def count_lines(path):
    with open(path, 'r') as f:
        return sum(1 for _ in f)


def main():
    parser = argparse.ArgumentParser(description='Validate HW4 Part 2 submission files.')
    parser.add_argument('--sql', required=True)
    parser.add_argument('--records', required=True)
    parser.add_argument('--test_nl', default=os.path.join(ROOT_DIR, 'data', 'test.nl'))
    args = parser.parse_args()

    sql_path = os.path.abspath(args.sql)
    record_path = os.path.abspath(args.records)
    test_nl_path = os.path.abspath(args.test_nl)

    if not os.path.exists(sql_path):
        raise FileNotFoundError(sql_path)
    if not os.path.exists(record_path):
        raise FileNotFoundError(record_path)
    if not os.path.exists(test_nl_path):
        raise FileNotFoundError(test_nl_path)

    sql_lines = count_lines(sql_path)
    test_lines = count_lines(test_nl_path)
    if sql_lines != test_lines:
        raise ValueError(f'SQL line count mismatch: sql={sql_lines}, test_nl={test_lines}')

    with open(record_path, 'rb') as f:
        records, error_msgs = pickle.load(f)

    if len(records) != test_lines:
        raise ValueError(f'Record count mismatch: records={len(records)}, test_nl={test_lines}')
    if len(error_msgs) != test_lines:
        raise ValueError(f'Error-msg count mismatch: error_msgs={len(error_msgs)}, test_nl={test_lines}')

    print('Validation passed')
    print(f'sql_path={sql_path}')
    print(f'record_path={record_path}')
    print(f'test_examples={test_lines}')
    print(f'nonempty_sql_errors={sum(bool(msg) for msg in error_msgs)}')


if __name__ == '__main__':
    main()
