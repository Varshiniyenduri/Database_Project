from itertools import combinations
import re
import pandas as pd


# Read the CSV file and functional dependencies file
students_data = pd.read_csv('students.csv')
print(students_data)
print('\n')

with open('students_data.txt', 'r') as f:
    dependency_lines = [line.strip() for line in f]

dependency_dict = {}
for line in dependency_lines:
    determinants, dependent = line.split(" -> ")
    determinants = determinants.split(", ")
    dependency_dict[tuple(determinants)] = dependent.split(", ")
print('Dependencies')
print(dependency_dict)
print('\n')

# Input from the user
selected_normal_form = input(
    "Choice of the highest normal form to reach (1: 1NF, 2: 2NF, 3: 3NF, B: BCNF, 4: 4NF, 5: 5NF): ")
if selected_normal_form in ["1", "2", "3", "4", "5"]:
    selected_normal_form = int(selected_normal_form)

# Determine the highest normal form of the relation
determine_high_normal_form = int(
    input('Find the highest normal form of the input table? (1: Yes, 2: No): '))
highest_normal_form = 'not normalized'

# Enter primary key
primary_key_values = input(
    "Enter the primary key values separated with commas: ").split(', ')
print('\n')

primary_key = tuple(primary_key_values)

def contains_comma(series):
    return series.str.contains(',').any()

def parse_csv_data(data):
    data = data.astype(str)
    columns_with_commas = [
        col for col in data.columns if contains_comma(data[col])]

    for col in columns_with_commas:
        data[col] = data[col].str.split(
            ',').apply(lambda x: [item.strip() for item in x])

    return data

students_data = parse_csv_data(students_data)

# Parsing the CSV file
def parse_csv(data):
    data = data.astype(str)
    columns_with_commas = [
        col for col in data.columns if contains_comma(data[col])]

    for col in columns_with_commas:
        data[col] = data[col].str.split(
            ',').apply(lambda x: [item.strip() for item in x])

    return data

parsed_data = parse_csv(students_data)

multivalued_dependencies = {}
if selected_normal_form != 'B' and selected_normal_form >= 4:
    with open('mvd.txt', 'r') as mvd_file:
        mvd_lines = [line.strip() for line in mvd_file]

    for mvd in mvd_lines:
        determinant, dependent = mvd.split(" ->> ")
        determinant = determinant.split(
            ", ") if ", " in determinant else [determinant]
        determinant_str = str(determinant)
        if determinant_str in multivalued_dependencies:
            multivalued_dependencies[determinant_str].append(dependent)
        else:
            multivalued_dependencies[determinant_str] = [dependent]

def is_list_or_set(item):
    return isinstance(item, (list, set))

def is_superkey(relation, determinant):
    grouped = relation.groupby(
        list(determinant)).size().reset_index(name='count')
    return not any(grouped['count'] > 1)

def powerset(s):
    length = len(s)
    for i in range(1 << length):
        yield [s[j] for j in range(length) if (i & (1 << j)) > 0]

def bcnf_decomposition(relation, dependencies):
    for determinant, dependents in dependencies.items():
        if set(determinant).issubset(relation.columns) and not is_superkey(relation, determinant):
            dependent_columns = list(determinant) + dependents
            new_relation1 = relation[dependent_columns].drop_duplicates()
            remaining_columns = list(set(relation.columns) - set(dependents))
            new_relation2 = relation[remaining_columns].drop_duplicates()
            return [new_relation1, new_relation2]
    return [relation]


def is_1nf(relation):
    if relation.empty:
        return False

    for column in relation.columns:
        unique_types = relation[column].apply(type).nunique()
        if unique_types > 1 or relation[column].apply(lambda x: isinstance(x, (list, dict, set))).any():
            return False

    return True


def is_2nf(primary_key, dependencies):
    for determinants, dependents in dependencies.items():
        superkey = all(attr in primary_key for attr in determinants)
        if superkey:
            for dependent in dependents:
                if dependent not in primary_key and dependent not in determinants:
                    return False
    return True


def is_3nf(relations, dependencies):
    for rel_name, rel in relations.items():
        attributes = set(rel.columns)
        non_prime_attributes = attributes - set(primary_key)

        for determinants, dependents in dependencies.items():
            if all(attr in non_prime_attributes for attr in determinants):
                for dependent in dependents:
                    if dependent in non_prime_attributes:
                        return False
    return True


def is_bcnf(relations, primary_key, dependencies):
    for rel_name, rel in relations.items():
        for determinants, dependents in dependencies.items():
            if set(determinants).issubset(rel.columns):
                if not is_superkey(rel, determinants):
                    return False
    return True


def is_4nf(relations, multivalued_dependencies):
    for rel_name, rel in relations.items():
        for determinants, dependents in multivalued_dependencies.items():
            for dependent in dependents:
                if isinstance(determinants, tuple):
                    determinant_cols = list(determinants)
                else:
                    determinant_cols = [determinants]

                if all(col in rel.columns for col in determinant_cols + [dependent]):
                    grouped = rel.groupby(determinant_cols)[
                        dependent].apply(set).reset_index()
                    if len(grouped) < len(rel):
                        print(
                            f"Violation of multi-valued dependencies: {determinants} ->> {dependent}")
                        return False

    return True
def is_5nf(relations):
    i = 0
    candidate_keys_dict = {}
    for rel_name, rel in relations.items():
        print(rel)
        user_input = input(f"Enter the candidate keys for relation '{rel_name}': ")
        print('\n')
        tuples = re.findall(r'\((.*?)\)', user_input)
        candidate_keys = [tuple(map(str.strip, t.split(','))) for t in tuples]
        candidate_keys_dict[i] = candidate_keys
        i += 1

    print(f'The Candidate Keys for given relations:')
    print(candidate_keys_dict)
    print('\n')

    j = 0
    for rel_name, rel in relations.items():
        candidate_keys = candidate_keys_dict[rel_name]
        j += 1

        data_tuples = [tuple(row) for row in rel.to_numpy()]

        def project(data, attributes):
            return {tuple(row[attr] for attr in attributes) for row in data}

        def is_superkey(attributes):
            for key in candidate_keys:
                if set(key).issubset(attributes):
                    return True
            return False

        for i in range(1, len(rel.columns)):
            for attrs in combinations(rel.columns, i):

                if is_superkey(attrs):
                    continue

                projected_data = project(data_tuples, attrs)
                complement_attrs = set(rel.columns) - set(attrs)
                complement_data = project(data_tuples, complement_attrs)
                joined_data = {(row1 + row2)
                               for row1 in projected_data for row2 in complement_data}
                if set(data_tuples) != joined_data:
                    print(f"Not satisfying 5NF in relation '{rel_name}' with attributes: {attrs}")
                    return False, candidate_keys_dict

    return True, candidate_keys_dict
def first_normal_form(relation):
    is_1nf_flag = is_1nf(relation)

    if is_1nf_flag:
        return relation, is_1nf_flag
    else:
        for column in relation.columns:
            if relation[column].apply(is_list_or_set).any():
                relation = relation.explode(column)

        return relation, is_1nf_flag

def second_normal_form(relation, primary_key, dependencies):
    relations = {}
    original_relation = relation
    is_2nf_flag = is_2nf(primary_key, dependencies)

    if is_2nf_flag:
        relations[tuple(primary_key)] = relation
        return relations, is_2nf_flag
    else:
        dependent_keys = list(dependencies.keys())
        for determinants, dependents in dependencies.items():
            modified_dependents = [
                dep + '_fk' if (dep,) in dependent_keys else dep for dep in dependents]

            columns = list(determinants) + dependents
            relations[tuple(determinants)] = relation[columns].drop_duplicates().reset_index(drop=True)

            rename_dict = {dep: modified_dep for dep, modified_dep in zip(dependents, modified_dependents)}
            relations[tuple(determinants)].rename(columns=rename_dict, inplace=True)

        junction_columns = []
        relation_name = ''
        for rel in relations:
            if set(rel).issubset(primary_key):
                relation_name += "_".join(rel)
                junction_columns.append(rel)

        if len(junction_columns) > 1:
            junction_cols = list(junction_columns)
            cols = [element for tup in junction_columns for element in tup]
            temp_df = original_relation[cols].drop_duplicates().reset_index(drop=True)

            renamed_cols = [col + '_fk' for col in cols]
            temp_df.columns = renamed_cols + [col for col in temp_df.columns if col not in cols]

            temp_df[relation_name] = range(1, len(temp_df) + 1)
            col_order = [relation_name] + renamed_cols
            temp_df = temp_df[col_order]
            relations[relation_name] = temp_df

        return relations, is_2nf_flag
def third_normal_form(relations, primary_key, dependencies):
    three_normal_form_rels = {}
    original_relations = relations
    is_3nf_flag = is_3nf(relations, dependencies)

    if is_3nf_flag:
        return relations, is_3nf_flag
    else:
        dependent_keys = list(dependencies.keys())
        for relation in relations:
            original_relation = relations[relation]
            for determinants, dependents in dependencies.items():
                modified_dependents = [
                    dep + '_fk' if (dep,) in dependent_keys else dep for dep in dependents]

                columns = list(determinants) + dependents
                three_normal_form_rels[tuple(determinants)] = relations[relation][columns].drop_duplicates(
                ).reset_index(drop=True)

                rename_dict = {dep: modified_dep for dep, modified_dep in zip(dependents, modified_dependents)}
                three_normal_form_rels[tuple(determinants)].rename(
                    columns=rename_dict, inplace=True)

        junction_columns = []

        relation_name = ''
        for relation in three_normal_form_rels:
            relation_name += "_".join(relation)
            junction_columns.append(relation)

        if len(junction_columns) > 1:
            junction_cols = list(junction_columns)
            cols = [element for tup in junction_columns for element in tup]
            temp_df = original_relations[cols].drop_duplicates(
            ).reset_index(drop=True)

            renamed_cols = [col + '_fk' for col in cols]
            temp_df.columns = renamed_cols + \
                              [col for col in temp_df.columns if col not in cols]

            temp_df[relation_name] = range(1, len(temp_df) + 1)
            col_order = [relation_name] + renamed_cols
            temp_df = temp_df[col_order]
            three_normal_form_rels[relation_name] = temp_df

        return three_normal_form_rels, is_3nf_flag

def bcnf(relations, primary_key, dependencies):
    relations = list(relations.values())
    bcnf_relations = []
    is_bcnf_flag = is_bcnf(relations, primary_key, dependencies)

    if is_bcnf_flag:
        return relations, is_bcnf_flag
    else:
        for relation in relations:
            bcnf_decomp_rel = bcnf_decomposition(
                relation, dependencies)
            if len(bcnf_decomp_rel) == 1:
                bcnf_relations.append(bcnf_decomp_rel)
            else:
                relations.extend(bcnf_decomp_rel)

    return bcnf_relations, is_bcnf_flag
def fourth_normal_form(relations, multivalued_dependencies):
    fourth_normal_form_rels = []
    is_four_nf_flag = is_4nf(relations, multivalued_dependencies)

    if is_four_nf_flag:
        return relations, is_four_nf_flag
    else:
        for relation in relations:
            for determinants, dependents in multivalued_dependencies.items():
                for dependent in dependents:
                    if isinstance(determinants, tuple):
                        determinant_cols = list(determinants)
                    else:
                        determinant_cols = [determinants]

                    if all(col in relation.columns for col in determinant_cols + [dependent]):
                        grouped = relation.groupby(determinant_cols)[
                            dependent].apply(set).reset_index()
                        if len(grouped) < len(relation):
                            # Decomposition
                            table_1 = relation[determinant_cols +
                                              [dependent]].drop_duplicates()
                            table_2 = relation[determinant_cols + [col for col in relation.columns if col not in [
                                dependent] + determinant_cols]].drop_duplicates()

                            fourth_normal_form_rels.extend([table_1, table_2])

                            break
                else:
                    continue
                break
            else:
                fourth_normal_form_rels.append(relation)

    if len(fourth_normal_form_rels) == len(relations):
        return fourth_normal_form_rels  # relations are in 4NF
    else:
        return fourth_normal_form(fourth_normal_form_rels, multivalued_dependencies)

def decompose_to_5nf(dataframe, candidate_keys):
    def project(df, attributes):
        return df[list(attributes)].drop_duplicates().reset_index(drop=True)

    # find whether the decomposition is lossless
    def is_lossless(df, df1, df2):
        common_columns = set(df1.columns) & set(df2.columns)
        if not common_columns:
            return False
        joined_df = pd.merge(df1, df2, how='inner', on=list(common_columns))
        return df.equals(joined_df)

    decomposed_tables = [dataframe]
# Check for each candidate key and then decompose the table
    for key in candidate_keys:
        new_tables = []
        for table in decomposed_tables:
            if set(key).issubset(set(table.columns)):
                table1 = project(table, key)
                remaining_columns = set(table.columns) - set(key)
                table2 = project(table, remaining_columns | set(key))

                # Check if the decomposition is lossless
                if is_lossless(table, table1, table2):
                    new_tables.extend([table1, table2])
                else:
                    new_tables.append(table)
            else:
                new_tables.append(table)
        decomposed_tables = new_tables

    return decomposed_tables

def fifth_normal_form(relations, primary_key, dependencies):
    fifth_normal_form_rels = []
    is_fifth_nf_flag, candidate_keys_dict = is_5nf(relations)

    if is_fifth_nf_flag:
        return relations, is_fifth_nf_flag
    else:
        i = 0
        for relation in relations:
            candidate_keys = candidate_keys_dict[i]
            i += 1
            decomposed_rels = decompose_to_5nf(relation, candidate_keys)
            fifth_normal_form_rels.append(decomposed_rels)

    return fifth_normal_form_rels, is_fifth_nf_flag

# Used to generate output as per the requirement
def output(data_type):
    """Change pandas data type to SQL data type."""
    if "int" in str(data_type):
        return "INT"
    elif "float" in str(data_type):
        return "FLOAT"
    elif "object" in str(data_type):
        return "VARCHAR(255)"
    elif "datetime" in str(data_type):
        return "DATETIME"
    else:
        return "TEXT"
def generate_sql_query_1NF(primary_keys, df):
    t_name = "_".join(primary_keys) + "_table"

    # Create SQL Query
    query = f"CREATE TABLE {t_name} (\n"

    # Iterate through columns to create query
    for col, dtype in zip(df.columns, df.dtypes):
        if col in primary_keys:
            query += f"  {col} {output(dtype)} PRIMARY KEY,\n"
        else:
            query += f"  {col} {output(dtype)},\n"

    query = query.rstrip(',\n') + "\n);"

    print(query)


def generate_sql_query_2_3(rels):
    for rel_name, rel in rels.items():
        primary_keys = rel_name
        primary_keys = (primary_keys,) if isinstance(
            primary_keys, str) else primary_keys
        t_name = "_".join(primary_keys) + '_table'

        # Create SQL Query
        query = f"CREATE TABLE {t_name} (\n"

        # Iterate through columns to create query
        for col, dtype in zip(rel.columns, rel.dtypes):
            if col in primary_keys:
                query += f"  {col} {output(dtype)} PRIMARY KEY,\n"
            elif '_fk' in col:
                query += f" FOREIGN KEY ({col}) REFERENCES {col.replace('_fk', '')}_table({col.replace('_fk', '')}),\n"
            else:
                query += f"  {col} {output(dtype)},\n"

        query = query.rstrip(',\n') + "\n);"

        print(query)
def generate_sql_query_BCNF_4_5(rels):
    for rel in rels:
        primary_key = rel.columns[0]
        t_name = f'{primary_key}_table'

        # Create SQL Query
        query = f"CREATE TABLE {t_name} (\n"

        # Iterate through columns to create query
        for col, dtype in zip(rel.columns, rel.dtypes):
            if col == primary_key:
                query += f"  {col} {output(dtype)} PRIMARY KEY,\n"
            elif '_fk' in col:
                query += f" FOREIGN KEY ({col}),\n"
            else:
                query += f"  {col} {output(dtype)},\n"

        query = query.rstrip(',\n') + "\n);"

        print(query)

def generate_sql_query_BCNF_4_5(rels):
    for rel in rels:
        primary_key = rel.columns[0]
        t_name = f'{primary_key}_table'

        # Create SQL Query
        query = f"CREATE TABLE {t_name} (\n"

        # Iterate through columns to create query
        for col, dtype in zip(rel.columns, rel.dtypes):
            if col == primary_key:
                query += f"  {col} {output(dtype)} PRIMARY KEY,\n"
            elif '_fk' in col:
                query += f" FOREIGN KEY ({col}),\n"
            else:
                query += f"  {col} {output(dtype)},\n"

        query = query.rstrip(',\n') + "\n);"

        print(query)

dependencies = {
    'Student ID': ['First Name', 'Last Name'],
    'Course': ['Professor', 'Professor Email', 'Course Start', 'Course End', 'Classroom'],
    # Add more dependencies as needed
}
# Define the target normal form
max_normal_form = 'B'  # Set this to the desired maximum normal form (or 'B' for BCNF)

# Rest of your code
if max_normal_form == 'B' or max_normal_form >= 1:
    onenf_table, one_flag = first_normal_form(students_data)
    if one_flag:
        high_normalform = 'The Highest Normal Form of the given table is: 1NF'

    if max_normal_form == 1:
        if one_flag:
            print('The table is already in 1NF')
            print('\n')

        generate_sql_query_1NF(primary_key, onenf_table)

if max_normal_form == 'B' or max_normal_form >= 2:
    twonf_tables, two_flag = second_normal_form(onenf_table, primary_key, dependencies)
    if one_flag and two_flag:
        high_normalform = 'The Highest Normal Form of the given table is: 2NF'

    if max_normal_form == 2:
        if two_flag and one_flag:
            print('The table is already in 2NF')
            print('\n')

        generate_sql_query_2_3(twonf_tables)
# Define the target normal form
max_normal_form = 'B'  # Set this to the desired maximum normal form (or 'B' for BCNF)

# Define a dictionary to map normal form values to their respective functions
normal_forms = {
    1: (first_normal_form, generate_sql_query_1NF),
    2: (second_normal_form, generate_sql_query_2_3),
    3: (third_normal_form, generate_sql_query_2_3),
    4: (bcnf, generate_sql_query_BCNF_4_5),
    5: (fourth_normal_form, generate_sql_query_BCNF_4_5),  # Note: 'B' is not included as it is handled separately
}

high_normalform = ''

# Handle BCNF ('B') as a special case
if max_normal_form == 'B':
    bcnf_tables, bcnf_flag = bcnf(table, primary_key, dependencies)

    if bcnf_flag:
        high_normalform = 'The Highest Normal Form of the given table is: BCNF'

    if bcnf_flag and max_normal_form == 'B':
        print('The table is already in BCNF')
        print('\n')

    generate_sql_query_BCNF_4_5(bcnf_tables)
else:
    # Iterate through normal forms up to the specified maximum
    max_normal_form = int(max_normal_form)  # Convert max_normal_form to an integer
    for nf in range(1, max_normal_form + 1):
        normal_form_func, sql_query_func = normal_forms[nf]
        table, flag = normal_form_func(students_data)

        if flag:
            high_normalform = f'The Highest Normal Form of the given table is: {nf}NF'

        if max_normal_form == nf:
            if flag:
                print(f'The table is already in {nf}NF')
                print('\n')
            sql_query_func(primary_key, table)

if determine_high_normal_form == 1:
    print('\n')
    print(high_normalform)
    print('\n')