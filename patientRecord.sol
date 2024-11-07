// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract MiniProject {
    struct Patient {
        string name;
        string recordType;
        string desc;
        uint256 date;
    }

    event recordAdded(address addr, string recordType, uint256 timestamp);
    event recordUpdated(address addr, string recordType, uint256 timestamp);

    mapping(address => Patient[]) public record;

    function addRecord(string memory _name, string memory _recordType, string memory _desc) public {
        Patient memory r1 = Patient({
            name: _name,
            recordType: _recordType,
            desc: _desc,
            date: block.timestamp
        });
        record[msg.sender].push(r1);
        emit recordAdded(msg.sender, _recordType, block.timestamp);
    }

    function updateRecord(uint256 index, string memory _name, string memory _recordType, string memory _desc) public {
        require(index < record[msg.sender].length, "Record does not exist"); 

        Patient storage r = record[msg.sender][index];  

        r.name = _name;
        r.recordType = _recordType;
        r.desc = _desc;
        r.date = block.timestamp; 

        emit recordUpdated(msg.sender, _recordType, block.timestamp);
    }
    function getRecord() public view returns (Patient[] memory) {
        return record[msg.sender];
    }
}
