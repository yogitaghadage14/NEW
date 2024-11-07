// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract Student {
    struct StudentData {
        uint256 id;
        string name;
        uint256 age;
        string course;
    }
    StudentData[] public students;
    uint256 public lastReceivedValue;

    event log(string func, uint gas);
    fallback() external payable{
        emit log("fallback",gasleft());
    }
    receive()external payable{
       emit log("receive",gasleft());
    }

    // Function to add a new student
    function addStudent(uint256 _id, string memory _name, uint256 _age, string memory _course) public {
        StudentData memory newStudent = StudentData({
            id: _id,
            name: _name,
            age: _age,
            course: _course
        });
        students.push(newStudent);
    }
    function getStudent(uint256 index) public view returns (uint256, string memory, uint256, string memory) {
        require(index < students.length, "Student not");
        StudentData memory student = students[index];
        return (student.id, student.name, student.age, student.course);
    }
    function getTotalStudents() public view returns (uint256) {
        return students.length;
    }
}


contract FallbackSender {
    function sendUsgTransfer(address payable addr) public payable {
        addr.transfer(msg.value); 
    }
}
