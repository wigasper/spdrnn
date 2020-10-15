

void test_routine() {

    std::vector<element_type> a_vals {1.5, 2.5, 3.5,
				      4.6, 5.6, 6.7};
    matrix a = std::make_tuple(a_vals, 3);

    print_matrix(a);

    tanh_e_wise(a);

    print_matrix(a); 

    std::vector<element_type> b_vals {1,2,3,4,5,6};
    matrix b = std::make_tuple(b_vals, 3);

    std::vector<element_type> c_vals {2,3,4,5,6,7};
    matrix c = std::make_tuple(c_vals, 3);

    print_matrix(b);
    print_matrix(c);
    add_in_place(b, c);

    print_matrix(b);
    print_matrix(c);

    std::cout << "testing appending !!!!!!\n";

    append_rows(b, c);
    std::cout<<"c appended to b\n";

    print_matrix(b);
    
    std::cout<<"c:\n";
    print_matrix(c);

    add_in_place(c, a);
    std::cout<<"c after adding a\n";
    print_matrix(c);

    std::cout<<"and then b:\n";
    print_matrix(b);

    std::cout<<"transposing b:\n";
    matrix d = transpose(b);
    print_matrix(d);
    
    std::vector<element_type> e_vals {1,2,3,4,5,6};
    matrix e = std::make_tuple(e_vals, 3);

    std::vector<element_type> f_vals {2,3,4,5,6,7};
    matrix f = std::make_tuple(f_vals, 3);

    print_matrix(e);
    print_matrix(f);
    
    matrix g = append_cols(e, f);
    print_matrix(g);

    pow_e_wise(g, 2);
    print_matrix(g);

    print_matrix(e);
    print_matrix(f);
    std::cout << "ewise mult of the two above\n";
    multiply(e, f);
    print_matrix(e);
}

