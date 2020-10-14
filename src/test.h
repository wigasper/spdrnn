

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
}
